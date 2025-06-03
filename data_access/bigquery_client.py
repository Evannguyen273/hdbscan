# data_access/bigquery_client.py
# BigQuery client for model versioning and data management
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib

from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Conflict

from config.config import load_config

class BigQueryClient:
    """
    BigQuery client for managing versioned model data and training datasets.
    Handles cumulative data storage and model registry operations.
    """
    
    def __init__(self, config=None):
        """Initialize BigQuery client with updated config system"""
        self.config = config if config is not None else load_config()
        
        # Initialize BigQuery client
        self._init_bigquery_client()
        
        # Get BigQuery configuration
        self.bq_config = self._get_bigquery_config()
        
        # Operation statistics
        self.operation_stats = {
            "queries_executed": 0,
            "datasets_created": 0,
            "tables_created": 0,
            "records_inserted": 0,
            "total_query_time": 0
        }
        
        logging.info("BigQuery client initialized for project: %s", self.project_id)
    
    def _init_bigquery_client(self):
        """Initialize BigQuery client"""
        bigquery_config = self.config.storage.bigquery
        
        self.project_id = bigquery_config.get('project_id')
        credentials_path = bigquery_config.get('credentials_path')
        
        if credentials_path:
            self.client = bigquery.Client.from_service_account_json(
                credentials_path, project=self.project_id
            )
        else:
            # Use default credentials (from environment)
            self.client = bigquery.Client(project=self.project_id)
    
    def _get_bigquery_config(self) -> Dict[str, Any]:
        """Get BigQuery configuration with defaults"""
        bq_config = self.config.storage.bigquery
        
        return {
            "dataset_id": bq_config.get('dataset_id', 'hdbscan_clustering'),
            "model_registry_table": bq_config.get('model_registry_table', 'model_registry'),
            "training_data_table": bq_config.get('training_data_table', 'training_data'),
            "cluster_results_table": bq_config.get('cluster_results_table', 'cluster_results'),
            "location": bq_config.get('location', 'US'),
            "enable_versioning": bq_config.get('enable_versioning', True),
            "retention_days": bq_config.get('retention_days', 730)  # 2 years
        }
    
    def create_dataset_if_not_exists(self) -> bool:
        """Create dataset if it doesn't exist"""
        try:
            dataset_id = f"{self.project_id}.{self.bq_config['dataset_id']}"
            
            try:
                self.client.get_dataset(dataset_id)
                logging.info("Dataset already exists: %s", dataset_id)
                return True
            except NotFound:
                # Create dataset
                dataset = bigquery.Dataset(dataset_id)
                dataset.location = self.bq_config['location']
                dataset.description = "HDBSCAN clustering model data and results"
                
                # Set retention policy
                retention_days = self.bq_config['retention_days']
                if retention_days > 0:
                    dataset.default_table_expiration_ms = retention_days * 24 * 60 * 60 * 1000
                
                dataset = self.client.create_dataset(dataset, timeout=30)
                
                self.operation_stats["datasets_created"] += 1
                logging.info("Created dataset: %s", dataset_id)
                return True
                
        except Exception as e:
            logging.error("Failed to create dataset: %s", str(e))
            return False
    
    def create_model_registry_table(self) -> bool:
        """Create model registry table for versioning"""
        try:
            table_id = f"{self.project_id}.{self.bq_config['dataset_id']}.{self.bq_config['model_registry_table']}"
            
            schema = [
                bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("model_hash", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("tech_center", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("training_date", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("model_metadata", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("performance_metrics", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("model_config", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("data_window_start", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("data_window_end", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("incidents_count", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("clusters_count", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("model_status", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            return self._create_table_with_schema(table_id, schema, "Model registry for HDBSCAN clustering")
            
        except Exception as e:
            logging.error("Failed to create model registry table: %s", str(e))
            return False
    
    def create_training_data_table(self, version: str) -> bool:
        """Create versioned training data table"""
        try:
            table_name = f"{self.bq_config['training_data_table']}_{version}"
            table_id = f"{self.project_id}.{self.bq_config['dataset_id']}.{table_name}"
            
            schema = [
                bigquery.SchemaField("incident_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("tech_center", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("original_text", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("processed_text", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("embedding", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("cluster_label", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("cluster_probability", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("domain", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("incident_date", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            return self._create_table_with_schema(
                table_id, schema, f"Training data for version {version}"
            )
            
        except Exception as e:
            logging.error("Failed to create training data table for version %s: %s", version, str(e))
            return False
    
    def create_cluster_results_table(self, version: str) -> bool:
        """Create versioned cluster results table"""
        try:
            table_name = f"{self.bq_config['cluster_results_table']}_{version}"
            table_id = f"{self.project_id}.{self.bq_config['dataset_id']}.{table_name}"
            
            schema = [
                bigquery.SchemaField("cluster_id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("tech_center", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("cluster_size", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("representative_incidents", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("cluster_keywords", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("cluster_centroid", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("cluster_description", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("domain", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("quality_score", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            return self._create_table_with_schema(
                table_id, schema, f"Cluster results for version {version}"
            )
            
        except Exception as e:
            logging.error("Failed to create cluster results table for version %s: %s", version, str(e))
            return False
    
    def get_training_data_window(self, start_date: str, end_date: str, tech_centers: List[str]) -> pd.DataFrame:
        """
        Fetch training data for specified date window and tech centers using configurable query
        """
        try:
            # Get query template and table from configuration
            query_template = self.config.bigquery.queries.training_data_window
            source_table = self.config.bigquery.tables.incident_source
            
            query = query_template.format(
                source_table=source_table,
                start_date=start_date,
                end_date=end_date
            )
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("tech_centers", "STRING", tech_centers)
                ]
            )
            
            df = self.client.query(query, job_config=job_config).to_dataframe()
            logging.info("Retrieved %d incidents for training window %s to %s", 
                        len(df), start_date, end_date)
            return df
            
        except Exception as e:
            logging.error("Failed to fetch training data: %s", str(e))
            raise
    
    def _create_table_with_schema(self, table_name: str, schema_fields: List[Dict]) -> bool:
        """Create BigQuery table with schema from configuration"""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
            
            # Convert schema dict to BigQuery schema fields
            schema = []
            for field in schema_fields:
                schema.append(bigquery.SchemaField(
                    field['name'], 
                    field['type'], 
                    mode=field.get('mode', 'NULLABLE'),
                    fields=field.get('fields', [])
                ))
            
            try:
                self.client.get_table(table_id)
                logging.info("Table already exists: %s", table_name)
                return True
            except NotFound:
                table = bigquery.Table(table_id, schema=schema)
                table = self.client.create_table(table)
                logging.info("Created table: %s", table_name)
                return True
                
        except Exception as e:
            logging.error("Failed to create table %s: %s", table_name, str(e))
            return False
    
    def register_model_version(self, model_metadata: Dict[str, Any]) -> bool:
        """Register model version using configurable insert query"""
        try:
            # Get query template and table from configuration
            query_template = self.config.bigquery.queries.model_registry_insert
            table = self.config.bigquery.tables.model_registry
            
            query = query_template.format(table=table)
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("model_version", "STRING", model_metadata["model_version"]),
                    bigquery.ScalarQueryParameter("tech_center", "STRING", model_metadata["tech_center"]),
                    bigquery.ScalarQueryParameter("model_type", "STRING", model_metadata["model_type"]),
                    bigquery.ScalarQueryParameter("training_data_start", "DATE", model_metadata["training_data_start"]),
                    bigquery.ScalarQueryParameter("training_data_end", "DATE", model_metadata["training_data_end"]),
                    bigquery.ScalarQueryParameter("blob_path", "STRING", model_metadata["blob_path"]),
                    bigquery.ScalarQueryParameter("created_timestamp", "TIMESTAMP", datetime.now()),
                    bigquery.ScalarQueryParameter("model_params", "JSON", json.dumps(model_metadata.get("model_params", {})))
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            query_job.result()  # Wait for completion
            
            logging.info("Registered model version: %s for tech center: %s", 
                        model_metadata["model_version"], model_metadata["tech_center"])
            return True
            
        except Exception as e:
            logging.error("Failed to register model version: %s", str(e))
            return False
    
    def get_latest_model_version(self, tech_center: str) -> Optional[Dict[str, Any]]:
        """Get latest model version for a tech center"""
        try:
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.bq_config['dataset_id']}.{self.bq_config['model_registry_table']}`
            WHERE tech_center = @tech_center 
                AND model_status = 'active'
            ORDER BY training_date DESC
            LIMIT 1
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("tech_center", "STRING", tech_center)
                ]
            )
            
            query_start = datetime.now()
            query_job = self.client.query(query, job_config=job_config)
            result = query_job.result()
            
            query_duration = datetime.now() - query_start
            self.operation_stats["queries_executed"] += 1
            self.operation_stats["total_query_time"] += query_duration.total_seconds()
            
            for row in result:
                return dict(row)
            
            return None
            
        except Exception as e:
            logging.error("Failed to get latest model version: %s", str(e))
            return None
      def get_training_data_window(self, tech_center: str, 
                               end_date: datetime, 
                               months_back: int = 24) -> pd.DataFrame:
        """Get training data for a specific window using configurable query"""
        try:
            start_date = end_date - timedelta(days=months_back * 30)
            
            # Use configurable query template and table reference
            query_template = self.config.bigquery.queries.training_data_window
            source_table = self.config.bigquery.tables.incident_source
            
            query = query_template.format(
                source_table=source_table,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("tech_center", "STRING", tech_center),
                    bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
                    bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date)
                ]
            )
            
            query_start = datetime.now()
            df = self.client.query(query, job_config=job_config).to_dataframe()
            
            query_duration = datetime.now() - query_start
            self.operation_stats["queries_executed"] += 1
            self.operation_stats["total_query_time"] += query_duration.total_seconds()
            
            logging.info("Retrieved %d incidents for %s from %s to %s", 
                        len(df), tech_center, start_date.date(), end_date.date())
            
            return df
            
        except Exception as e:
            logging.error("Failed to get training data window: %s", str(e))
            return pd.DataFrame()
      def store_training_data(self, version: str, training_data: pd.DataFrame) -> bool:
        """Store training data for a version"""
        try:
            # Use configuration-driven table references
            table_name = f"{self.config.bigquery.tables.training_data}_{version}"
            table_id = f"{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.{table_name}"
            
            # Add created_at timestamp
            training_data = training_data.copy()
            training_data['created_at'] = datetime.now()
            
            # Load data to BigQuery
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",  # Replace table contents
                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
            )
            
            job = self.client.load_table_from_dataframe(
                training_data, table_id, job_config=job_config
            )
            job.result()  # Wait for completion
            
            self.operation_stats["records_inserted"] += len(training_data)
            logging.info("Stored %d training records for version %s", len(training_data), version)
            return True
            
        except Exception as e:
            logging.error("Failed to store training data: %s", str(e))
            return False
      def store_cluster_results(self, version: str, cluster_results: List[Dict]) -> bool:
        """Store cluster results for a version"""
        try:
            # Use configuration-driven table references
            table_name = f"{self.config.bigquery.tables.cluster_results}_{version}"
            table_id = f"{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.{table_name}"
            
            # Add created_at timestamp to each result
            for result in cluster_results:
                result['created_at'] = datetime.now()
            
            # Insert data using proper table reference
            errors = self.client.insert_rows_json(
                self.client.get_table(table_id), cluster_results
            )
            errors = self.client.insert_rows_json(
                self.client.get_table(table_id), cluster_results
            )
            
            if errors:
                logging.error("Failed to store cluster results: %s", errors)
                return False
            
            self.operation_stats["records_inserted"] += len(cluster_results)
            logging.info("Stored %d cluster results for version %s", len(cluster_results), version)
            return True
            
        except Exception as e:
            logging.error("Failed to store cluster results: %s", str(e))
            return False
    
    def cleanup_old_versions(self, retention_days: int = None) -> bool:
        """Clean up old model versions and data"""
        try:
            retention_days = retention_days or self.bq_config['retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Mark old models as archived
            query = f"""
            UPDATE `{self.project_id}.{self.bq_config['dataset_id']}.{self.bq_config['model_registry_table']}`
            SET model_status = 'archived', updated_at = CURRENT_TIMESTAMP()
            WHERE training_date < @cutoff_date AND model_status = 'active'
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("cutoff_date", "TIMESTAMP", cutoff_date)
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            result = query_job.result()
            
            logging.info("Archived models older than %d days", retention_days)
            return True
            
        except Exception as e:
            logging.error("Failed to cleanup old versions: %s", str(e))
            return False
    
    def generate_model_hash(self, model_data: Dict[str, Any]) -> str:
        """Generate hash for model versioning"""
        # Create a consistent hash based on model parameters and data characteristics
        hash_input = {
            "config": model_data.get("config", {}),
            "training_date": model_data.get("training_date", "").split("T")[0],  # Date only
            "incidents_count": model_data.get("incidents_count", 0),
            "tech_center": model_data.get("tech_center", "")
        }
        
        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()[:8]
    
    def get_versioned_table_name(self, version: str, model_hash: str) -> str:
        """Get versioned table name"""
        return f"{version}_{model_hash}"
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get BigQuery operation statistics"""
        return self.operation_stats.copy()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate BigQuery configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        if not self.project_id:
            validation_results["errors"].append("BigQuery project_id not configured")
            validation_results["valid"] = False
        
        if not self.bq_config["dataset_id"]:
            validation_results["errors"].append("BigQuery dataset_id not configured")
            validation_results["valid"] = False
        
        # Test connection
        try:
            self.client.get_dataset(f"{self.project_id}.information_schema")
        except Exception as e:
            validation_results["errors"].append(f"BigQuery connection failed: {str(e)}")
            validation_results["valid"] = False
        
        return validation_results
    
    def reset_statistics(self):
        """Reset operation statistics"""
        self.operation_stats = {
            "queries_executed": 0,
            "datasets_created": 0,
            "tables_created": 0,
            "records_inserted": 0,
            "total_query_time": 0
        }
        logging.info("BigQuery client statistics reset")
    async def insert_training_log(self, log_entry: Dict) -> bool:
        """Insert training log entry into BigQuery"""
        try:
            table_id = self.config.bigquery.tables.training_logs
            
            # Add timestamp if not provided
            if 'timestamp' not in log_entry:
                log_entry['timestamp'] = datetime.now()
            
            # Validate required fields
            required_fields = ['run_id', 'log_level', 'message']
            for field in required_fields:
                if field not in log_entry:
                    raise ValueError(f"Missing required field: {field}")
            
            return await self.insert_rows(table_id, [log_entry])
            
        except Exception as e:
            logging.error("Failed to insert training log: %s", str(e))
            return False
    
    async def get_watermark(self, pipeline_name: str, tech_center: str) -> Optional[datetime]:
        """Get watermark for a specific pipeline and tech center"""
        try:
            table_id = self.config.bigquery.tables.watermarks
            
            query = f"""
            SELECT last_processed_timestamp
            FROM `{table_id}`
            WHERE pipeline_name = @pipeline_name 
            AND tech_center = @tech_center
            ORDER BY updated_at DESC
            LIMIT 1
            """
            
            job_config = self.create_query_job_config()
            job_config.query_parameters = [
                bigquery.ScalarQueryParameter("pipeline_name", "STRING", pipeline_name),
                bigquery.ScalarQueryParameter("tech_center", "STRING", tech_center)
            ]
            
            df = await self.execute_query(query, job_config)
            
            if len(df) > 0 and not pd.isna(df['last_processed_timestamp'].iloc[0]):
                return df['last_processed_timestamp'].iloc[0]
            
            return None
            
        except Exception as e:
            logging.error("Failed to get watermark: %s", str(e))
            return None
    
    async def update_watermark(self, pipeline_name: str, tech_center: str, 
                             timestamp: datetime, last_processed_id: str = None) -> bool:
        """Update watermark for a specific pipeline and tech center"""
        try:
            table_id = self.config.bigquery.tables.watermarks
            
            watermark_entry = {
                'pipeline_name': pipeline_name,
                'tech_center': tech_center,
                'last_processed_timestamp': timestamp,
                'last_processed_id': last_processed_id,
                'updated_at': datetime.now()
            }
            
            # Use MERGE to upsert the watermark
            query = f"""
            MERGE `{table_id}` T
            USING (
                SELECT 
                    @pipeline_name as pipeline_name,
                    @tech_center as tech_center,
                    @last_processed_timestamp as last_processed_timestamp,
                    @last_processed_id as last_processed_id,
                    @updated_at as updated_at
            ) S
            ON T.pipeline_name = S.pipeline_name AND T.tech_center = S.tech_center
            WHEN MATCHED THEN
                UPDATE SET 
                    last_processed_timestamp = S.last_processed_timestamp,
                    last_processed_id = S.last_processed_id,
                    updated_at = S.updated_at
            WHEN NOT MATCHED THEN
                INSERT (pipeline_name, tech_center, last_processed_timestamp, last_processed_id, updated_at)
                VALUES (S.pipeline_name, S.tech_center, S.last_processed_timestamp, S.last_processed_id, S.updated_at)
            """
            
            job_config = self.create_query_job_config()
            job_config.query_parameters = [
                bigquery.ScalarQueryParameter("pipeline_name", "STRING", pipeline_name),
                bigquery.ScalarQueryParameter("tech_center", "STRING", tech_center),
                bigquery.ScalarQueryParameter("last_processed_timestamp", "TIMESTAMP", timestamp),
                bigquery.ScalarQueryParameter("last_processed_id", "STRING", last_processed_id),
                bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", datetime.now())
            ]
            
            query_job = self.client.query(query, job_config=job_config)
            query_job.result()  # Wait for completion
            
            logging.info("Updated watermark for %s - %s to %s", pipeline_name, tech_center, timestamp)
            return True
            
        except Exception as e:
            logging.error("Failed to update watermark: %s", str(e))
            return False
    
    async def create_operational_tables(self) -> bool:
        """Create operational tables (training_logs, watermarks) if they don't exist"""
        try:
            # Create training_logs table
            training_logs_schema = self.config.bigquery.schemas.get('training_logs', [])
            if training_logs_schema:
                training_logs_success = self.create_table_if_not_exists(
                    self.config.bigquery.tables.training_logs,
                    training_logs_schema
                )
            else:
                training_logs_success = False
                logging.error("training_logs schema not found in configuration")
            
            # Create watermarks table
            watermarks_schema = self.config.bigquery.schemas.get('watermarks', [])
            if watermarks_schema:
                watermarks_success = self.create_table_if_not_exists(
                    self.config.bigquery.tables.watermarks,
                    watermarks_schema
                )
            else:
                watermarks_success = False
                logging.error("watermarks schema not found in configuration")
            
            return training_logs_success and watermarks_success
            
        except Exception as e:
            logging.error("Failed to create operational tables: %s", str(e))
            return False