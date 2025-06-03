"""
Schema definitions for HDBSCAN Pipeline
Provides Pydantic models for data validation and documentation
"""

from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class IncidentSchema(BaseModel):
    """Schema for incident data"""
    incident_number: str
    description: str
    created_date: datetime
    tech_center: str
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None
    
    @validator('incident_number')
    def validate_incident_number(cls, v):
        if not v or not v.strip():
            raise ValueError("Incident number cannot be empty")
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Description must be at least 10 characters")
        return v.strip()

class PreprocessedIncidentSchema(BaseModel):
    """Schema for preprocessed incident data with embeddings"""
    incident_number: str
    tech_center: str
    description_summary: str
    embedding: List[float]
    preprocessing_version: str
    created_timestamp: datetime
    
    @validator('embedding')
    def validate_embedding(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Embedding cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain only numeric values")
        return v

class TrainingDataSchema(BaseModel):
    """Schema for training dataset"""
    incident_number: str
    tech_center: str
    description_summary: str
    embedding: List[float]
    training_version: str
    created_timestamp: datetime

class ClusterResultSchema(BaseModel):
    """Schema for clustering results"""
    incident_number: str
    tech_center: str
    cluster_id: int
    confidence_score: Optional[float] = None
    model_version: str
    prediction_timestamp: datetime
    domain_group: Optional[str] = None
    
    @validator('cluster_id')
    def validate_cluster_id(cls, v):
        if v < -1:  # -1 is noise cluster in HDBSCAN
            raise ValueError("Cluster ID must be >= -1")
        return v

class PredictionResultSchema(BaseModel):
    """Schema for prediction results"""
    incident_number: str
    predicted_cluster: int
    confidence_score: float
    tech_center: str
    model_version: str
    prediction_timestamp: datetime
    domain_group: Optional[str] = None
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1")
        return v

class ModelRegistrySchema(BaseModel):
    """Schema for model registry entries"""
    model_version: str
    tech_center: str
    model_type: str  # 'hdbscan', 'umap', 'domain_grouper'
    training_data_start: datetime
    training_data_end: datetime
    blob_path: str
    created_timestamp: datetime
    model_params: Optional[Dict[str, Any]] = None
    cluster_count: Optional[int] = None
    silhouette_score: Optional[float] = None
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ['hdbscan', 'umap', 'domain_grouper', 'preprocessor']
        if v not in allowed_types:
            raise ValueError(f"Model type must be one of: {allowed_types}")
        return v

class ModelMetricsSchema(BaseModel):
    """Schema for model performance metrics"""
    model_version: str
    tech_center: str
    cluster_count: int
    noise_ratio: float
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    training_incidents_count: int
    training_duration_seconds: float
    
    @validator('noise_ratio')
    def validate_noise_ratio(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Noise ratio must be between 0 and 1")
        return v

class ValidationResultSchema(BaseModel):
    """Schema for text validation results"""
    incident_number: str
    is_valid: bool
    failure_reason: Optional[str] = None
    estimated_tokens: Optional[int] = None
    text_length: Optional[int] = None

class EmbeddingBatchSchema(BaseModel):
    """Schema for embedding generation batch"""
    batch_id: str
    incidents: List[IncidentSchema]
    total_count: int
    valid_count: int
    failed_count: int
    processing_timestamp: datetime
    
    @validator('valid_count', 'failed_count')
    def validate_counts(cls, v, values):
        if 'total_count' in values and v > values['total_count']:
            raise ValueError("Count cannot exceed total count")
        return v

class TrainingLogSchema(BaseModel):
    """Schema for training log entries"""
    run_id: str
    timestamp: datetime
    pipeline_stage: Optional[str] = None
    tech_center: Optional[str] = None
    model_version: Optional[str] = None
    log_level: str
    message: str
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WatermarkSchema(BaseModel):
    """Schema for processing watermarks"""
    pipeline_name: str
    tech_center: str
    last_processed_timestamp: Optional[datetime] = None
    last_processed_id: Optional[str] = None
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class VersionedTrainingResultSchema(BaseModel):
    """Schema for versioned training results (clustering_predictions_{version}_{hash})"""
    incident_number: str
    cluster_id: int
    cluster_label: Optional[str] = None
    domain_id: Optional[int] = None
    domain_name: Optional[str] = None
    umap_x: Optional[float] = None
    umap_y: Optional[float] = None
    tech_center: str
    model_version: str
    confidence_score: Optional[float] = None
    created_timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class IncidentPredictionSchema(BaseModel):
    """Schema for live incident predictions"""
    incident_id: str
    predicted_cluster_id: int
    predicted_cluster_label: Optional[str] = None
    confidence_score: Optional[float] = None
    predicted_domain_id: Optional[int] = None
    predicted_domain_name: Optional[str] = None
    tech_center: str
    prediction_timestamp: datetime
    model_table_used: Optional[str] = None
    blob_model_path: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Utility functions for schema validation
def validate_incident_batch(incidents: List[Dict]) -> List[ValidationResultSchema]:
    """Validate a batch of incidents and return validation results"""
    results = []
    
    for incident in incidents:
        try:
            IncidentSchema(**incident)
            results.append(ValidationResultSchema(
                incident_number=incident.get('incident_number', 'unknown'),
                is_valid=True
            ))
        except Exception as e:
            results.append(ValidationResultSchema(
                incident_number=incident.get('incident_number', 'unknown'),
                is_valid=False,
                failure_reason=str(e)
            ))
    
    return results

def get_bigquery_schema(schema_name: str) -> List[Dict]:
    """Get BigQuery schema definition for a given schema"""
    schema_mapping = {
        'incidents': [
            {'name': 'incident_number', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'description', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'created_date', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'tech_center', 'type': 'STRING', 'mode': 'REQUIRED'},
        ],
        'preprocessed_incidents': [
            {'name': 'incident_number', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'tech_center', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'description_summary', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'embedding', 'type': 'REPEATED', 'mode': 'NULLABLE',
             'fields': [{'name': 'value', 'type': 'FLOAT'}]},
            {'name': 'preprocessing_version', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'created_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
        ],
        'cluster_results': [
            {'name': 'incident_number', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'tech_center', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'cluster_id', 'type': 'INTEGER', 'mode': 'REQUIRED'},
            {'name': 'confidence_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'model_version', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'prediction_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'domain_group', 'type': 'STRING', 'mode': 'NULLABLE'},
        ],
        'model_registry': [
            {'name': 'model_version', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'tech_center', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'model_type', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'training_data_start', 'type': 'DATE', 'mode': 'REQUIRED'},
            {'name': 'training_data_end', 'type': 'DATE', 'mode': 'REQUIRED'},
            {'name': 'blob_path', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'created_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'model_params', 'type': 'JSON', 'mode': 'NULLABLE'},
            {'name': 'cluster_count', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'silhouette_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
        ]
    }
    
    return schema_mapping.get(schema_name, [])