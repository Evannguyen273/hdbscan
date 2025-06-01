import logging
import time
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import json
import os
from typing import Optional

class BigQueryClient:
    def __init__(self, config):
        self.config = config
        self.project_id = config.get('bigquery', {}).get('project_id', '')
        
        # Handle service account from environment variable
        service_account_info = config.get('bigquery', {}).get('service_account_key_path', {})
        
        if isinstance(service_account_info, dict):
            # Already parsed JSON from config.py
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
        else:
            # Parse if still string
            service_account_info = json.loads(os.getenv('SERVICE_ACCOUNT_KEY_PATH'))
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            
        self.client = bigquery.Client(project=self.project_id, credentials=credentials)
        
        # Store table references from config
        self.tables = config.get('bigquery', {}).get('tables', {})
        logging.info(f'Connected to BigQuery project: {self.project_id}')
    
    def get_table_id(self, table_name: str) -> str:
        """Get full table ID from config"""
        if table_name in self.tables:
            return self.tables[table_name]
        return table_name
    
    def run_query(self, query: str, max_retries: int = 3) -> pd.DataFrame:
        """Execute a query with retry logic"""
        retry_count = 0
        backoff_time = 2
        
        while retry_count <= max_retries:
            try:
                query_job = self.client.query(query)
                result = query_job.result().to_dataframe(create_bqstorage_client=False)
                logging.info(f"Query executed successfully, returned {len(result)} rows")
                return result
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logging.error(f"Query failed after {max_retries} attempts: {e}")
                    return pd.DataFrame()
                
                logging.warning(f"Query attempt {retry_count} failed: {e}. Retrying in {backoff_time}s")
                time.sleep(backoff_time)
                backoff_time *= 2
    
    def get_raw_incidents(self, tech_center: str = None, limit: int = None) -> pd.DataFrame:
        """Get raw incidents from the incidents table"""
        table_id = self.get_table_id('raw_incidents')
        
        query = f"""
        SELECT 
            number,
            sys_created_on,
            short_description,
            description,
            tech_center,
            state,
            priority,
            created_by
        FROM `{table_id}`
        """
        
        if tech_center:
            query += f" WHERE tech_center = '{tech_center}'"
        
        query += " ORDER BY sys_created_on DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.run_query(query)
    
    def get_team_services(self) -> pd.DataFrame:
        """Get team services mapping"""
        table_id = self.get_table_id('team_services')
        
        query = f"""
        SELECT *
        FROM `{table_id}`
        """
        
        return self.run_query(query)
    
    def get_problems(self, tech_center: str = None) -> pd.DataFrame:
        """Get problems from the problems table"""
        table_id = self.get_table_id('problems')
        
        query = f"""
        SELECT 
            number,
            sys_created_on,
            short_description,
            description,
            tech_center,
            state
        FROM `{table_id}`
        """
        
        if tech_center:
            query += f" WHERE tech_center = '{tech_center}'"
        
        return self.run_query(query)
    
    def get_preprocessed_incidents(self, tech_center: str = None, after_timestamp: str = None) -> pd.DataFrame:
        """Get preprocessed incidents with embeddings"""
        table_id = self.get_table_id('preprocessed_incidents')
        
        query = f"""
        SELECT 
            number,
            sys_created_on,
            combined_incidents_summary,
            embedding,
            tech_center,
            processed_timestamp
        FROM `{table_id}`
        """
        
        conditions = []
        if tech_center:
            conditions.append(f"tech_center = '{tech_center}'")
        
        if after_timestamp:
            conditions.append(f"sys_created_on > '{after_timestamp}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY sys_created_on DESC"
        
        return self.run_query(query)
    
    def get_watermark(self, tech_center: str, pipeline_type: str) -> Optional[str]:
        """Get the last processed timestamp for watermark-based processing"""
        table_id = self.get_table_id('preprocessing_watermarks')
        
        query = f"""
        SELECT max_processed_timestamp
        FROM `{table_id}`
        WHERE tech_center = '{tech_center}' 
        AND pipeline_type = '{pipeline_type}'
        ORDER BY updated_timestamp DESC
        LIMIT 1
        """
        
        result = self.run_query(query)
        if not result.empty:
            return result.iloc[0]['max_processed_timestamp']
        return None
    
    def update_watermark(self, tech_center: str, pipeline_type: str, timestamp: str) -> bool:
        """Update the watermark for processed records"""
        table_id = self.get_table_id('preprocessing_watermarks')
        
        # Use MERGE to insert or update
        query = f"""
        MERGE `{table_id}` T
        USING (
            SELECT 
                '{tech_center}' as tech_center,
                '{pipeline_type}' as pipeline_type,
                '{timestamp}' as max_processed_timestamp,
                CURRENT_TIMESTAMP() as updated_timestamp
        ) S
        ON T.tech_center = S.tech_center AND T.pipeline_type = S.pipeline_type
        WHEN MATCHED THEN
            UPDATE SET 
                max_processed_timestamp = S.max_processed_timestamp,
                updated_timestamp = S.updated_timestamp
        WHEN NOT MATCHED THEN
            INSERT (tech_center, pipeline_type, max_processed_timestamp, updated_timestamp)
            VALUES (S.tech_center, S.pipeline_type, S.max_processed_timestamp, S.updated_timestamp)
        """
        
        try:
            job = self.client.query(query)
            job.result()
            logging.info(f"Updated watermark for {tech_center}/{pipeline_type}: {timestamp}")
            return True
        except Exception as e:
            logging.error(f"Failed to update watermark: {e}")
            return False
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, 
                      write_disposition: str = "WRITE_APPEND") -> bool:
        """Save DataFrame to BigQuery table"""
        try:
            table_id = self.get_table_id(table_name)
            write_disp = getattr(bigquery.WriteDisposition, write_disposition)
            job_config = bigquery.LoadJobConfig(write_disposition=write_disp)
            
            job = self.client.load_table_from_dataframe(
                df, table_id, job_config=job_config, timeout=300
            )
            job.result()
            logging.info(f"Successfully saved {len(df)} records to {table_id}")
            return True
        except Exception as e:
            logging.error(f"Error saving to BigQuery table {table_name}: {e}")
            return False
    
    def save_preprocessed_incidents(self, df: pd.DataFrame) -> bool:
        """Save preprocessed incidents with embeddings"""
        return self.save_dataframe(df, 'preprocessed_incidents')
    
    def save_clustering_predictions(self, df: pd.DataFrame) -> bool:
        """Save clustering prediction results"""
        return self.save_dataframe(df, 'clustering_predictions')