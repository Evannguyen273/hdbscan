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
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to BigQuery using credentials from config"""
        try:
            if os.path.isfile(self.config.bigquery.service_account_key_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.bigquery.service_account_key_path
                )
            else:
                json_key = json.loads(self.config.bigquery.service_account_key_path)
                credentials = service_account.Credentials.from_service_account_info(json_key)
            
            self.client = bigquery.Client(
                credentials=credentials, 
                project=self.config.bigquery.project_id
            )
            logging.info('Connected to BigQuery')
        except Exception as e:
            raise ConnectionError(f"Failed to connect to BigQuery: {e}")
    
    def run_query(self, query: str, max_retries: int = 3) -> pd.DataFrame:
        """Execute a query with retry logic"""
        retry_count = 0
        backoff_time = 2
        
        while retry_count <= max_retries:
            try:
                query_job = self.client.query(query)
                return query_job.result().to_dataframe(create_bqstorage_client=False)
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logging.error(f"Query failed after {max_retries} attempts: {e}")
                    return pd.DataFrame()
                
                logging.warning(f"Query attempt {retry_count} failed: {e}. Retrying in {backoff_time}s")
                time.sleep(backoff_time)
                backoff_time *= 2
    
    def save_dataframe(self, df: pd.DataFrame, table_id: str, 
                      write_disposition: str = "WRITE_APPEND") -> bool:
        """Save DataFrame to BigQuery table"""
        try:
            write_disp = getattr(bigquery.WriteDisposition, write_disposition)
            job_config = bigquery.LoadJobConfig(write_disposition=write_disp)
            
            job = self.client.load_table_from_dataframe(
                df, table_id, job_config=job_config, timeout=300
            )
            job.result()
            logging.info(f"Successfully saved {len(df)} records to {table_id}")
            return True
        except Exception as e:
            logging.error(f"Error saving to BigQuery: {e}")
            return False