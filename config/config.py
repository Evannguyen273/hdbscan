import os
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

@dataclass
class AzureConfig:
    openai_endpoint: str
    openai_api_key: str
    openai_embedding_endpoint: str
    openai_embedding_key: str
    chat_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    api_version: str = "2024-02-01"
    blob_connection_string: Optional[str] = None
    container_name: str = "prediction-artifacts"

@dataclass
class BigQueryConfig:
    project_id: str
    embeddings_dataset: str
    results_dataset: str
    service_account_key_path: str

@dataclass
class ClusteringConfig:
    min_cluster_size: int = 25
    min_samples: int = 5
    umap_n_components: int = 50
    umap_n_neighbors: int = 100
    umap_min_dist: float = 0.1
    max_domains: int = 20

@dataclass
class EmbeddingConfig:
    entity_weight: float = 0.0
    action_weight: float = 0.0
    semantic_weight: float = 1.0
    batch_size: int = 25

@dataclass
class PipelineConfig:
    save_to_local: bool = True
    result_path: str = "results"
    use_checkpointing: bool = True

@dataclass
class Config:
    azure: AzureConfig
    bigquery: BigQueryConfig
    clustering: ClusteringConfig
    embedding: EmbeddingConfig
    pipeline: PipelineConfig
    tech_centers: List[str]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file with environment variable substitution"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Substitute environment variables
        config_data = cls._substitute_env_vars(config_data)
        
        return cls(
            azure=AzureConfig(**config_data['azure']),
            bigquery=BigQueryConfig(**config_data['bigquery']),
            clustering=ClusteringConfig(**config_data.get('clustering', {})),
            embedding=EmbeddingConfig(**config_data.get('embedding', {})),
            pipeline=PipelineConfig(**config_data.get('pipeline', {})),
            tech_centers=config_data.get('tech_centers', [])
        )
    
    @staticmethod
    def _substitute_env_vars(data: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(data, dict):
            return {k: Config._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Config._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        return data

def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load configuration from file"""
    return Config.from_yaml(config_path)