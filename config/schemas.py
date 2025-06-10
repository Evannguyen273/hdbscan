"""
Schema definitions for HDBSCAN Training Pipeline
Updated to match config.yaml structure and training pipeline requirements
"""

from google.cloud import bigquery
from typing import List, Dict, Any

def get_schema_from_config(config: Dict[str, Any], schema_name: str) -> List[bigquery.SchemaField]:
    """
    Convert schema definition from config to BigQuery SchemaField objects
    
    Args:
        config: Configuration dictionary
        schema_name: Name of the schema in config.bigquery.schemas
        
    Returns:
        List of BigQuery SchemaField objects
    """
    schema_config = config.get('bigquery', {}).get('schemas', {}).get(schema_name, [])
    
    schema_fields = []
    for field_config in schema_config:
        # Handle repeated fields (like embeddings)
        if field_config.get('mode') == 'REPEATED':
            if 'fields' in field_config:
                # Nested repeated field (like embedding with subfields)
                subfields = []
                for subfield in field_config['fields']:
                    subfields.append(bigquery.SchemaField(
                        subfield['name'], 
                        subfield['type'], 
                        mode=subfield.get('mode', 'NULLABLE')
                    ))
                schema_fields.append(bigquery.SchemaField(
                    field_config['name'],
                    'RECORD',
                    mode='REPEATED',
                    fields=subfields
                ))
            else:
                # Simple repeated field (like float array)
                schema_fields.append(bigquery.SchemaField(
                    field_config['name'],
                    field_config['type'],
                    mode='REPEATED'
                ))
        else:
            # Regular field
            schema_fields.append(bigquery.SchemaField(
                field_config['name'],
                field_config['type'],
                mode=field_config.get('mode', 'NULLABLE')
            ))
    
    return schema_fields

# Training Cycle Metadata Schema (for multi-tech center orchestration)
TRAINING_CYCLE_METADATA_SCHEMA = [
    bigquery.SchemaField("training_cycle_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("tech_center", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("tech_center_slug", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("training_month", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("training_year", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("training_completed_date", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("umap_artifact_path", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("hdbscan_artifact_path", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("metadata_artifact_path", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("cluster_results_table", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("embeddings_source_table", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("model_version", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("model_hash", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("clusters_count", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("domains_count", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("incidents_count", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("silhouette_score", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("noise_ratio", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("training_status", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_timestamp", "TIMESTAMP", mode="REQUIRED"),
]

# Cluster Results Schema (versioned tables: cluster_results_{version}_{hash})
CLUSTER_RESULTS_SCHEMA = [
    bigquery.SchemaField("number", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("sys_created_on", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("combined_incidents_summary", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("tech_center", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("cluster_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("cluster_label", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("cluster_description", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("domain_id", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("domain_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("umap_x", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("umap_y", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("model_hash", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("prediction_timestamp", "TIMESTAMP", mode="REQUIRED"),
]

# Preprocessed Incidents Schema (with embeddings stored here)
PREPROCESSED_INCIDENTS_SCHEMA = [
    bigquery.SchemaField("number", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("sys_created_on", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("combined_incidents_summary", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
    bigquery.SchemaField("created_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("processing_version", "STRING", mode="REQUIRED"),
]

# Model Registry Schema
MODEL_REGISTRY_SCHEMA = [
    bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("tech_center", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("model_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("training_data_start", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("training_data_end", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("blob_path", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("created_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("model_params", "JSON", mode="NULLABLE"),
    bigquery.SchemaField("cluster_count", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("silhouette_score", "FLOAT", mode="NULLABLE"),
]

# Training Logs Schema
TRAINING_LOGS_SCHEMA = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("pipeline_stage", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("tech_center", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("model_version", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("log_level", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("message", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("details", "JSON", mode="NULLABLE"),
]

# Watermarks Schema
WATERMARKS_SCHEMA = [
    bigquery.SchemaField("pipeline_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("tech_center", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("last_processed_timestamp", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("last_processed_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
]

# Incident Predictions Schema (for live predictions)
INCIDENT_PREDICTIONS_SCHEMA = [
    bigquery.SchemaField("incident_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("predicted_cluster_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("predicted_cluster_label", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("confidence_score", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("predicted_domain_id", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("predicted_domain_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("tech_center", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("prediction_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("model_table_used", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("blob_model_path", "STRING", mode="NULLABLE"),
]

# Preprocessing Watermarks Schema
PREPROCESSING_WATERMARKS_SCHEMA = [
    bigquery.SchemaField("preprocessed_rows", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("time_trigger", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("run_details", "JSON", mode="NULLABLE"),
]

# Schema mapping for easy access
SCHEMA_MAPPING = {
    'training_cycle_metadata': TRAINING_CYCLE_METADATA_SCHEMA,
    'cluster_results': CLUSTER_RESULTS_SCHEMA,
    'preprocessed_incidents': PREPROCESSED_INCIDENTS_SCHEMA,
    'model_registry': MODEL_REGISTRY_SCHEMA,
    'training_logs': TRAINING_LOGS_SCHEMA,
    'watermarks': WATERMARKS_SCHEMA,
    'incident_predictions': INCIDENT_PREDICTIONS_SCHEMA,
    'preprocessing_watermarks': PREPROCESSING_WATERMARKS_SCHEMA,
}

def get_schema_by_name(schema_name: str) -> List[bigquery.SchemaField]:
    """
    Get BigQuery schema by name
    
    Args:
        schema_name: Name of the schema
        
    Returns:
        List of BigQuery SchemaField objects
        
    Raises:
        ValueError: If schema name is not found
    """
    if schema_name not in SCHEMA_MAPPING:
        available_schemas = list(SCHEMA_MAPPING.keys())
        raise ValueError(f"Schema '{schema_name}' not found. Available: {available_schemas}")
    
    return SCHEMA_MAPPING[schema_name]

def create_table_if_not_exists(client: bigquery.Client, table_id: str, schema_name: str, 
                              partition_field: str = None, cluster_fields: List[str] = None) -> bool:
    """
    Create BigQuery table if it doesn't exist
    
    Args:
        client: BigQuery client
        table_id: Full table ID (project.dataset.table)
        schema_name: Schema name from SCHEMA_MAPPING
        partition_field: Field to partition by (optional)
        cluster_fields: Fields to cluster by (optional)
        
    Returns:
        True if table was created or already exists
    """
    try:
        # Check if table exists
        client.get_table(table_id)
        return True
    except Exception:
        # Table doesn't exist, create it
        try:
            schema = get_schema_by_name(schema_name)
            table = bigquery.Table(table_id, schema=schema)
            
            # Add partitioning if specified
            if partition_field:
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field=partition_field
                )
            
            # Add clustering if specified
            if cluster_fields:
                table.clustering_fields = cluster_fields
            
            table = client.create_table(table)
            return True
            
        except Exception as e:
            print(f"Failed to create table {table_id}: {e}")
            return False