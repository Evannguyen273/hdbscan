# data_access/bigquery_client.py
# BigQuery client for model versioning and data management
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib
import os

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
        """Initialize BigQuery client with credentials"""
        try:
            # Access bigquery directly
            bigquery_config = self.config.bigquery

            # Get project ID from config
            self.project_id = bigquery_config.project_id

            # Create credentials from service account info
            from google.oauth2 import service_account

            # Handle service account key - could be a dict (parsed JSON) or a string path
            service_account_info = bigquery_config.service_account_key_path

            if isinstance(service_account_info, dict):
                # Use the service account info directly
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                logging.info("Using service account credentials from config JSON")
            else:
                # Attempt to load from file path
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        service_account_info,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    logging.info(f"Using service account credentials from file: {service_account_info}")
                except Exception as e:
                    logging.warning(f"Could not load service account from path: {e}")
                    logging.info("Falling back to application default credentials")
                    # Fall back to default credentials
                    credentials = None

            # Initialize BigQuery client with explicit credentials
            self.client = bigquery.Client(
                project=self.project_id,
                credentials=credentials
            )

            # Store dataset ID - use project_id if dataset not found
            self.dataset_id = getattr(bigquery_config, 'dataset', self.project_id)

            logging.info("BigQuery client initialized for project: %s", self.project_id)

        except Exception as e:
            logging.error("Failed to initialize BigQuery client: %s", e)
            raise

    def _get_bigquery_config(self) -> Dict[str, Any]:
        """Get BigQuery configuration with defaults"""
        # Fix: Access bigquery directly
        bq_config = self.config.bigquery

        # Use a default value for cluster_results if not provided
        default_cluster_results = f"{self.project_id}.{getattr(bq_config, 'dataset', 'hdbscan_clustering')}.cluster_results"

        return {
            "dataset_id": getattr(bq_config, 'dataset', self.project_id) if hasattr(bq_config, 'dataset') else 'hdbscan_clustering',
            "model_registry_table": bq_config.tables.model_registry if hasattr(bq_config, 'tables') else 'model_registry',
            "training_data_table": bq_config.tables.training_data if hasattr(bq_config, 'tables') else 'training_data',
            "cluster_results_table": bq_config.tables.cluster_results if hasattr(bq_config, 'tables') and bq_config.tables.cluster_results else default_cluster_results,
            "location": getattr(bq_config, 'location', 'US') if hasattr(bq_config, 'location') else 'US',
            "enable_versioning": True,
            "retention_days": 730  # 2 years
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
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED")            ]

            table_id = f"{self.project_id}.{self.bq_config['dataset_id']}.{self.bq_config['model_registry_table']}"
            return self._create_table_with_schema(table_id, schema)

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
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")            ]

            return self._create_table_with_schema(table_id, schema)

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
            return self._create_table_with_schema(table_id, schema)

        except Exception as e:
            logging.error("Failed to create cluster results table for version %s: %s", version, str(e))
            return False

    def get_training_data_window(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch training data for specified date window without tech center filtering
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

            # No query parameters needed - removed tech_centers filter
            df = self.client.query(query).to_dataframe()

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

            # Log parameters for debugging
            logging.info(f"Querying data for tech_center='{tech_center}', from {start_date} to {end_date}")

            # Use a direct query approach for better control and debugging
            source_table = self.config.bigquery.tables.incident_source

            # Create a direct query with proper date format handling
            # Note: In BigQuery, TIMESTAMP format like '2024-01-01T00:00:00' works directly with sys_created_on
            direct_query = f"""
            SELECT
                number,
                sys_created_on,
                description,
                short_description,
                business_service
            FROM `{source_table}`
            WHERE sys_created_on >= '{start_date.isoformat()}'
            AND sys_created_on <= '{end_date.isoformat()}'
            """

            # Log the raw query for debugging
            logging.info(f"Executing raw query: {direct_query}")

            # Execute the basic query without tech center filtering first
            query_job = self.client.query(direct_query)
            all_df = query_job.to_dataframe()

            if all_df.empty:
                logging.warning(f"No data found for any tech center in the specified date range: {start_date} to {end_date}")
                return pd.DataFrame()

            # Report all unique business_service values found
            if 'business_service' in all_df.columns:
                unique_services = all_df['business_service'].unique().tolist()
                logging.info(f"Found {len(unique_services)} unique business services: {unique_services}")

            # Now filter for the requested tech center in memory, trying different variations
            filtered_dfs = []

            # Try exact match
            exact_match = all_df[all_df['business_service'] == tech_center]
            if not exact_match.empty:
                filtered_dfs.append(("Exact match", exact_match))

            # Try simplified name (remove "BT-TC-" prefix)
            simplified_tc = tech_center
            if tech_center.startswith("BT-TC-"):
                simplified_tc = tech_center[6:]  # Remove "BT-TC-" prefix
                simplified_match = all_df[all_df['business_service'].str.contains(simplified_tc, case=False, na=False)]
                if not simplified_match.empty:
                    filtered_dfs.append(("Simplified match", simplified_match))

            # Try with "Development" instead of "Development &"
            if "Development &" in tech_center:
                dev_match = all_df[all_df['business_service'].str.contains("Development", case=False, na=False)]
                if not dev_match.empty:
                    filtered_dfs.append(("Development match", dev_match))

            # If we found any matches
            if filtered_dfs:
                # Use the first successful match approach
                match_type, df = filtered_dfs[0]
                logging.info(f"Found {len(df)} rows using {match_type}")
                return df
            else:
                logging.warning(f"No data found for tech center '{tech_center}' in {len(all_df)} total rows")
                return pd.DataFrame()

        except Exception as e:
            logging.error(f"Failed to get training data window: {e}", exc_info=True)
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

            if errors:
                logging.error("Failed to store cluster results: %s", errors)
                return False

            self.operation_stats["records_inserted"] += len(cluster_results)
            logging.info("Stored %d cluster results for version %s", len(cluster_results), version)
            return True
        except Exception as e:
            logging.error("Failed to store cluster results: %s", str(e))
            return False

    def store_preprocessed_incidents(self, df: pd.DataFrame, overwrite: bool = False) -> bool:
        """
        Store preprocessed incidents to BigQuery

        Args:
            df: DataFrame containing preprocessed incidents
            overwrite: If True, overwrite the existing table. If False (default), append to the table.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            table_name = self.config.bigquery.tables.preprocessed_incidents

            # Log detailed information about the destination
            project, dataset, table = self._parse_table_reference(table_name)
            write_mode = "TRUNCATE" if overwrite else "APPEND"
            logging.info(f"Storing {len(df)} preprocessed incidents to {project}.{dataset}.{table} (mode: {write_mode})")

            # Check if dataset exists, create if not
            try:
                self.client.get_dataset(dataset)
            except NotFound:
                logging.warning(f"Dataset {dataset} not found. Attempting to create it...")
                dataset_ref = bigquery.Dataset(f"{project}.{dataset}")
                dataset_ref.location = "US"  # Set your preferred location
                self.client.create_dataset(dataset_ref, exists_ok=True)

            # Ensure all expected columns are present
            schema = self._get_table_schema('preprocessed_incidents')
            expected_columns = [field.name for field in schema]
            for column in expected_columns:
                if column not in df.columns:
                    logging.warning(f"Column '{column}' missing from DataFrame, adding empty column")
                    if column == 'embedding':
                        df[column] = [[] for _ in range(len(df))]
                    else:
                        df[column] = None

            # Load data into BigQuery with appropriate write_disposition
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                write_disposition="WRITE_TRUNCATE" if overwrite else "WRITE_APPEND"
            )

            job = self.client.load_table_from_dataframe(
                df, table_name, job_config=job_config
            )

            # Wait for the job to complete
            job.result()

            logging.info(f"Loaded {len(df)} rows to {table_name} ({write_mode} mode)")
            return True

        except Exception as e:
            logging.error(f"Failed to store preprocessed incidents: {e}")
            return False

    def _parse_table_reference(self, table_ref: str) -> tuple:
        """Parse a table reference string into project, dataset, and table components"""
        parts = table_ref.split('.')
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            return self.config.bigquery.project_id, parts[0], parts[1]
        else:
            return self.config.bigquery.project_id, "default", parts[0]

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

    async def get_watermark_for_preprocessing(self, pipeline_name: str, tech_center: str) -> Optional[datetime]:
        """
        Get the most recent incident timestamp from preprocessed_incidents table

        Returns the sys_created_on timestamp of the newest incident that was successfully processed
        in the previous pipeline run.
        """
        try:
            # Get the table ID from configuration
            table_id = self.config.bigquery.tables.preprocessed_incidents

            # Get the query template from config and format it
            query_template = self.config.bigquery.queries.get_watermark_for_preprocessing
            query = query_template.format(table_id=table_id, tech_center_filter="")

            # Execute the query directly without additional parameters
            df = self.client.query(query).to_dataframe()

            if len(df) > 0 and not pd.isna(df['last_processed_timestamp'].iloc[0]):
                return df['last_processed_timestamp'].iloc[0]

            return None

        except Exception as e:
            logging.error(f"Failed to get watermark timestamp for preprocessing: {e}")
            return None

    async def update_watermark(self, pipeline_name: str, tech_center: str,
                             timestamp: datetime, last_processed_id: str = None) -> bool:
        """
        Update watermark for a specific pipeline and tech center

        Args:
            pipeline_name: Name of the pipeline (e.g., "preprocessing")
            tech_center: Tech center being processed, or "all" for all tech centers
            timestamp: The sys_created_on timestamp of the most recent incident processed
            last_processed_id: Optional ID of the last processed incident

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            table_id = self.config.bigquery.tables.watermarks

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

            job_config = bigquery.QueryJobConfig()
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
                training_logs_success = self._create_table_with_schema(
                    self.config.bigquery.tables.training_logs,
                    training_logs_schema
                )
            else:
                training_logs_success = False
                logging.error("training_logs schema not found in configuration")

            # Create watermarks table
            watermarks_schema = self.config.bigquery.schemas.get('watermarks', [])
            if watermarks_schema:
                watermarks_success = self._create_table_with_schema(
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

    async def create_preprocessing_watermarks_table(self) -> bool:
        """Create the preprocessing watermarks table if it doesn't exist"""
        try:
            # Use the correct table name from config
            watermark_table = self.config.bigquery.tables.watermarks
            watermarks_schema = self.config.bigquery.schemas.get('preprocessing_watermarks', [])

            if not watermarks_schema:
                logging.error("preprocessing_watermarks schema not found in configuration")
                return False

            # Extract project, dataset, and table from the table reference
            project, dataset, table = self._parse_table_reference(watermark_table)
            table_id = f"{project}.{dataset}.{table}"

            # Check if table exists
            try:
                self.client.get_table(table_id)
                logging.info(f"Watermarks table already exists: {table_id}")
                return True
            except NotFound:
                # Convert schema fields to BigQuery schema
                schema = []
                for field in watermarks_schema:
                    schema.append(bigquery.SchemaField(
                        field['name'],
                        field['type'],
                        mode=field.get('mode', 'NULLABLE'),
                        fields=field.get('fields', [])
                    ))

                # Create dataset with location if it doesn't exist
                dataset_id = f"{project}.{dataset}"
                try:
                    self.client.get_dataset(dataset_id)
                except NotFound:
                    # Create dataset with specified location
                    dataset_obj = bigquery.Dataset(dataset_id)
                    dataset_obj.location = "europe-west1"
                    self.client.create_dataset(dataset_obj)
                    logging.info(f"Created dataset with location europe-west1: {dataset_id}")

                # Create table without setting location directly
                table_obj = bigquery.Table(table_id, schema=schema)
                self.client.create_table(table_obj)
                logging.info(f"Created preprocessing watermarks table: {table_id}")
                return True

        except Exception as e:
            logging.error(f"Failed to create preprocessing watermarks table: {e}")
            return False

    async def update_preprocessing_watermark(self, preprocessed_rows: int, timestamp: datetime, run_details: Dict = None, overwrite: bool = False) -> bool:
        """
        Update the preprocessing watermark table with the number of rows processed and provided timestamp

        Args:
            preprocessed_rows: Total number of rows processed in this run
            timestamp: The timestamp to use for time_trigger
            run_details: Additional details about the run
            overwrite: If True, overwrite existing watermarks. If False (default), append a new record.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use the correct table name from config
            watermark_table = self.config.bigquery.tables.watermarks

            # Create table if it doesn't exist
            table_exists = await self.create_preprocessing_watermarks_table()
            if not table_exists:
                logging.error("Failed to create or verify watermark table")
                return False

            # Standardize datetime format in run_details to use ISO 8601 with 'T' separator
            if run_details:
                # Create a deep copy to avoid modifying the original
                import copy
                formatted_run_details = copy.deepcopy(run_details)

                # Format all date fields to use 'T' separator ISO format
                for key, value in formatted_run_details.items():
                    if isinstance(value, datetime):
                        # Convert directly to ISO format with 'T' separator
                        formatted_run_details[key] = value.strftime('%Y-%m-%dT%H:%M:%S')
                    elif isinstance(value, str) and (
                        ' UTC' in value or
                        (' ' in value and len(value) >= 19 and value[10] == ' ')
                    ):
                        # Handle string dates with spaces and possible UTC suffix
                        try:
                            # Try to parse and reformat the date string
                            date_str = value.replace(' UTC', '')  # Remove UTC suffix if present
                            parsed_date = datetime.strptime(date_str.strip(), '%Y-%m-%d %H:%M:%S')
                            formatted_run_details[key] = parsed_date.strftime('%Y-%m-%dT%H:%M:%S')
                        except Exception:
                            # If parsing fails, keep the original value
                            pass
            else:
                formatted_run_details = run_details

            # Convert run_details to JSON
            run_details_json = json.dumps(formatted_run_details) if formatted_run_details else None

            # Format the timestamp for BigQuery (use ISO format with 'T' separator)
            timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S')

            # If overwrite is True, delete existing records first
            if overwrite:
                delete_query = f"DELETE FROM `{watermark_table}` WHERE 1=1"
                self.client.query(delete_query).result()
                logging.info(f"Overwrite mode: Deleted existing watermark records")

            # Insert query (same for both modes)
            insert_query = f"""
            INSERT INTO `{watermark_table}`
            (preprocessed_rows, time_trigger, run_details)
            VALUES
            (@preprocessed_rows, TIMESTAMP '{timestamp_str}', @run_details)
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("preprocessed_rows", "INTEGER", preprocessed_rows),
                    bigquery.ScalarQueryParameter("run_details", "JSON", run_details_json)
                ]
            )

            # Execute the query
            query_job = self.client.query(insert_query, job_config=job_config, location="europe-west1")
            query_job.result()

            mode = "overwrite" if overwrite else "append"
            logging.info(f"Updated preprocessing watermark with {preprocessed_rows} rows at {timestamp_str} (mode: {mode})")
            return True

        except Exception as e:
            logging.error(f"Failed to update preprocessing watermark: {e}")
            return False

    async def get_latest_preprocessing_watermark(self) -> Tuple[Optional[datetime], Optional[int]]:
        """
        Get the most recent preprocessing watermark timestamp and row count

        Returns:
            Tuple of (timestamp, row_count) or (None, None) if no watermark exists
        """
        try:
            # Use the correct table name from config (watermarks instead of preprocessing_watermarks)
            watermark_table = self.config.bigquery.tables.watermarks

            # Modified query to order by time_trigger
            query = f"""
            SELECT time_trigger, preprocessed_rows
            FROM `{watermark_table}`
            ORDER BY time_trigger DESC
            LIMIT 1
            """

            df = self.client.query(query).to_dataframe()

            if not df.empty:
                timestamp = df['time_trigger'].iloc[0]
                row_count = df['preprocessed_rows'].iloc[0]
                return timestamp, row_count

            return None, None

        except Exception as e:
            logging.error(f"Failed to get latest preprocessing watermark: {e}")
            return None, None

    async def get_latest_preprocessing_watermark_with_details(self) -> Optional[Dict]:
        """
        Get the complete latest preprocessing watermark row including run_details

        Returns:
            Dictionary with complete watermark information or None if no watermark exists
        """
        try:
            watermark_table = self.config.bigquery.tables.watermarks

            # Add a more explicit log to show the query
            print(f"🔍 DEBUG: Querying watermark table: {watermark_table}")

            query = f"""
            SELECT time_trigger, preprocessed_rows, run_details
            FROM `{watermark_table}`
            ORDER BY time_trigger DESC
            LIMIT 1
            """

            print(f"🔍 DEBUG: Executing watermark query: {query}")
            df = self.client.query(query).to_dataframe()

            if not df.empty:
                # Convert row to dict
                row_dict = df.iloc[0].to_dict()
                print(f"🔍 DEBUG: Found watermark row: time_trigger={row_dict.get('time_trigger')}")
                return row_dict
            else:
                print("🔍 DEBUG: No watermark rows found in table")

            return None

        except Exception as e:
            logging.error(f"Failed to get latest preprocessing watermark with details: {e}")
            return None

    def _get_table_schema(self, schema_name: str) -> List[bigquery.SchemaField]:
        """
        Get BigQuery schema definition for a given table

        Args:
            schema_name: Name of the schema from config (e.g., 'preprocessed_incidents')

        Returns:
            List of BigQuery SchemaField objects
        """
        try:
            # Get schema from configuration
            schema_definition = self.config.bigquery.schemas.get(schema_name, [])

            if not schema_definition:
                logging.error(f"Schema '{schema_name}' not found in configuration")
                return []

            # Convert schema dict to BigQuery schema fields
            schema = []
            for field in schema_definition:
                field_name = field.get('name')
                field_type = field.get('type')
                field_mode = field.get('mode', 'NULLABLE')
                field_fields = field.get('fields', [])

                # Special handling for REPEATED RECORD type (for nested fields)
                if field_type == 'REPEATED' and field_fields:
                    # Create nested fields
                    nested_fields = []
                    for nested_field in field_fields:
                        nested_fields.append(
                            bigquery.SchemaField(
                                name=nested_field.get('name'),
                                field_type=nested_field.get('type'),
                                mode=nested_field.get('mode', 'NULLABLE')
                            )
                        )

                    schema.append(
                        bigquery.SchemaField(
                            name=field_name,
                            field_type='RECORD',
                            mode='REPEATED',
                            fields=nested_fields
                        )
                    )
                else:
                    schema.append(
                        bigquery.SchemaField(
                            name=field_name,
                            field_type=field_type,
                            mode=field_mode
                        )
                    )

            return schema

        except Exception as e:
            logging.error(f"Failed to get schema for '{schema_name}': {e}")
            return []