# Configuration Manager for HDBSCAN Pipeline
# Updated for cumulative training with versioned storage architecture

import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, validator

class Config:
    """Configuration manager for consolidated config.yaml"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load environment variables
        self._load_environment()
          # Determine config file
        if config_path is None:
            config_path = self._get_default_config_path()
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.config_path = config_path
        
        # Apply environment variable substitution
        self.config = self._substitute_env_vars(self.config)
        
        # Validate configuration
        self._validate_config()
        
        logging.info("Configuration loaded from: %s", config_path)
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logging.info("Loaded environment variables from %s", env_file)
        else:
            logging.warning("No .env file found, using system environment variables")
    
    def _get_default_config_path(self) -> str:
        """Get default config path using config.yaml"""
        config_dir = Path(__file__).parent
        main_config = config_dir / 'config.yaml'
        if main_config.exists():
            return str(main_config)
        raise FileNotFoundError("Configuration file not found: config.yaml")
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]  # Remove ${ and }
            env_value = os.getenv(env_var)
            if env_value is None:
                logging.warning("Environment variable %s not found", env_var)
                return obj
              # Handle JSON strings
            if env_var == 'SERVICE_ACCOUNT_KEY_PATH':
                try:
                    return json.loads(env_value)
                except json.JSONDecodeError:
                    return env_value
            
            return env_value
        else:
            return obj
    
    def _validate_config(self):
        """Validate that required configuration sections exist"""
        required_sections = ['bigquery', 'azure', 'clustering', 'training', 'prediction']
        
        for section in required_sections:
            if section not in self.config:
                logging.warning("Configuration section '%s' not found, using defaults", section)
          # Validate required environment variables
        required_env_vars = [
            'AZURE_OPENAI_ENDPOINT',
            'OPENAI_API_KEY',
            'BLOB_CONNECTION_STRING'
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logging.error("Missing required environment variables: %s", missing_vars)
            raise ValueError("Missing required environment variables: %s" % missing_vars)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def bigquery(self):
        """Get BigQuery configuration"""
        return BigQueryConfig(self.config.get('bigquery', {}))
    
    @property
    def azure(self):
        """Get Azure configuration"""
        return AzureConfig(self.config.get('azure', {}))
    
    @property
    def clustering(self):
        """Get clustering configuration"""
        return ClusteringConfig(self.config.get('clustering', {}))
    
    @property
    def training(self):
        """Get training configuration"""
        return TrainingConfig(self.config.get('training', {}))
    
    @property
    def prediction(self):
        """Get prediction configuration"""
        return PredictionConfig(self.config.get('prediction', {}))
    
    @property
    def tech_centers(self):
        """Get tech centers configuration"""
        if 'tech_centers' in self.config:
            tech_centers_config = self.config['tech_centers']
            if isinstance(tech_centers_config, dict):
                # New format with primary/additional
                primary = tech_centers_config.get('primary', [])
                additional = tech_centers_config.get('additional', [])
                
                # Convert primary list to names
                primary_names = []
                for tc in primary:
                    if isinstance(tc, dict) and 'name' in tc:
                        primary_names.append(tc['name'])
                    else:
                        primary_names.append(str(tc))
                
                return primary_names + additional
            else:
                # Old format - simple list
                return tech_centers_config
        return []
    
    @property
    def monitoring(self):
        """Get monitoring configuration"""
        return MonitoringConfig(**self.config.get('monitoring', {}))
    
    @property
    def cost_optimization(self):
        """Get cost optimization configuration"""
        return CostOptimizationConfig(**self.config.get('cost_optimization', {}))
    
    @property
    def performance(self):
        """Get performance configuration"""
        return PerformanceConfig(**self.config.get('performance', {}))
    
    @property
    def security(self):
        """Get security configuration"""  
        return SecurityConfig(**self.config.get('security', {}))
    
    @property
    def tech_centers_config(self):
        """Get tech centers configuration with validation"""
        return TechCentersConfig(**self.config.get('tech_centers', {}))
    
    @property
    def logging_config(self):
        """Get logging configuration"""
        return LoggingConfig(**self.config.get('logging', {}))

class BigQueryTableConfig(BaseModel):
    """Configuration for BigQuery tables with validation"""
    incidents: str
    team_services: str
    problems: str
    incident_source: str
    preprocessed_incidents: str
    training_results_template: str
    training_data: str
    predictions: str
    model_registry: str
    training_logs: str  # Added for operational logging
    watermarks: str     # Added for processing checkpoints
    
    @validator('*')
    def validate_table_names(cls, v):
        if not v or not v.strip():
            raise ValueError("Table name cannot be empty")
        return v

class BigQueryQueriesConfig(BaseModel):
    """Configuration for BigQuery SQL queries"""
    training_data_window: str
    model_registry_insert: str
    cluster_results_insert: str
    
    @validator('*')
    def validate_queries(cls, v):
        if not v or not v.strip():
            raise ValueError("Query template cannot be empty")
        return v

class BigQueryConfig(BaseModel):
    """BigQuery configuration with validation"""
    project_id: str
    service_account_key_path: str
    tables: BigQueryTableConfig
    queries: BigQueryQueriesConfig
    schemas: Dict[str, List[Dict]]
    
    @validator('project_id')
    def validate_project_id(cls, v):
        if not v or not v.strip():
            raise ValueError("BigQuery project_id cannot be empty")
        return v

class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration with validation"""
    endpoint: str
    api_key: str
    api_version: str
    deployment_name: str
    embedding_endpoint: str
    embedding_api_version: str
    embedding_key: str
    embedding_model: str
    
    @validator('endpoint', 'embedding_endpoint')
    def validate_endpoints(cls, v):
        if not v or not v.startswith('https://'):
            raise ValueError("Endpoint must be a valid HTTPS URL")
        return v

class AzureBlobConfig(BaseModel):
    """Azure Blob Storage configuration"""
    connection_string: str
    container_name: str
    structure: Dict[str, str]

class AzureConfig(BaseModel):
    """Azure configuration container"""
    openai: AzureOpenAIConfig
    blob_storage: AzureBlobConfig

class ValidationConfig(BaseModel):
    """Text validation configuration"""
    max_embedding_tokens: int = 8191
    min_text_length: int = 10
    max_text_length: int = 32000
    
    @validator('max_embedding_tokens')
    def validate_token_limit(cls, v):
        if v <= 0:
            raise ValueError("max_embedding_tokens must be positive")
        return v

class ClusteringConfig:
    """Clustering configuration wrapper"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @property
    def hdbscan(self) -> Dict[str, Any]:
        return self.config.get('hdbscan', {})
    
    @property
    def umap(self) -> Dict[str, Any]:
        return self.config.get('umap', {})
    
    @property
    def domain_grouping(self) -> Dict[str, Any]:
        """Get domain grouping configuration"""
        return self.config.get('domain_grouping', {
            'enabled': True,
            'max_domains_per_tech_center': 20,
            'min_incidents_per_domain': 5,
            'similarity_threshold': 0.7
        })
    
    @property
    def max_domains(self) -> int:
        """Get max domains - backward compatibility"""
        return self.domain_grouping.get('max_domains_per_tech_center', 20)
    
    @property
    def min_incidents_per_domain(self) -> int:
        """Get min incidents per domain - backward compatibility"""
        return self.domain_grouping.get('min_incidents_per_domain', 5)

class TrainingConfig:
    """Training configuration wrapper for cumulative training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @property
    def schedule(self) -> Dict[str, Any]:
        """Get training schedule configuration"""
        return self.config.get('schedule', {
            'frequency': 'semi_annual',
            'months': [6, 12],
            'training_window_months': 24
        })
    
    @property
    def frequency(self) -> str:
        """Get training frequency"""
        return self.schedule.get('frequency', 'semi_annual')
    
    @property
    def training_window_months(self) -> int:
        """Get training window in months (cumulative approach)"""
        return self.schedule.get('training_window_months', 24)
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get training parameters"""
        return self.config.get('parameters', {})
    
    @property
    def versioning(self) -> Dict[str, Any]:
        """Get model versioning configuration"""
        return self.config.get('versioning', {
            'version_format': '{year}_q{quarter}',
            'hash_algorithm': 'sha256',
            'hash_length': 8
        })
    
    @property
    def processing(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.config.get('processing', {
            'parallel_tech_centers': True,
            'max_workers': 4,
            'timeout_hours': 6,
            'batch_size': 1000,
            'max_incidents_per_training': 100000
        })

class PredictionConfig:
    """Prediction configuration wrapper for real-time classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @property
    def schedule(self) -> Dict[str, Any]:
        """Get prediction schedule configuration"""
        return self.config.get('schedule', {
            'frequency_minutes': 120,
            'batch_size': 500,
            'timeout_minutes': 30
        })
    
    @property
    def frequency_minutes(self) -> int:
        """Get prediction frequency in minutes"""
        return self.schedule.get('frequency_minutes', 120)
    
    @property
    def batch_size(self) -> int:
        """Get prediction batch size"""
        return self.schedule.get('batch_size', 500)
    
    @property
    def model_loading(self) -> Dict[str, Any]:
        """Get model loading configuration"""
        return self.config.get('model_loading', {
            'cache_models': True,
            'cache_ttl_hours': 24,
            'version_strategy': 'latest',
            'fallback_version': '2024_q4',
            'preload_models': True
        })
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get prediction parameters"""
        return self.config.get('parameters', {
            'min_confidence_score': 0.3,
            'high_confidence_threshold': 0.8,
            'max_distance_to_cluster': 2.0,
            'enable_domain_prediction': True
        })

# Add new Pydantic models for expanded configuration sections

class MonitoringConfig(BaseModel):
    """Configuration for monitoring and alerting"""
    metrics: Dict[str, List[str]]
    alerts: Dict[str, bool]

class CostOptimizationConfig(BaseModel):
    """Configuration for cost optimization strategies"""
    storage: Dict[str, bool]
    training: Dict[str, bool]
    queries: Dict[str, Any]

class PerformanceConfig(BaseModel):
    """Configuration for performance and scaling"""
    memory: Dict[str, Any]
    parallel: Dict[str, Any]
    caching: Dict[str, Any]

class SecurityConfig(BaseModel):
    """Configuration for security and governance"""
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    audit_logging: bool = True
    rbac_enabled: bool = True
    service_account_rotation: bool = True
    data_retention_days: int = 730
    pii_handling: str = "anonymized"

class TechCenterConfig(BaseModel):
    """Configuration for tech centers"""
    name: str
    slug: str
    min_incidents: int

class TechCentersConfig(BaseModel):
    """Configuration for all tech centers"""
    primary: List[TechCenterConfig]
    additional: List[str]

class LoggingConfig(BaseModel):
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: Dict[str, Dict[str, Any]]

# Global configuration instance
_config_instance = None

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration, using cached instance if available"""
    global _config_instance
    
    if _config_instance is None or config_path is not None:
        _config_instance = Config(config_path)
    
    return _config_instance

def get_config() -> Config:
    """Get the current configuration instance"""
    if _config_instance is None:
        return load_config()
    return _config_instance

# Legacy functions for backward compatibility
def load_yaml_config(config_path: str = None) -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    config = load_config(config_path)
    return config.config

def get_tech_centers() -> list:
    """Get list of tech centers"""
    config = get_config()
    return config.tech_centers

def get_current_quarter() -> str:
    """Get current quarter based on current month"""
    import datetime
    current_month = datetime.datetime.now().month
    
    if current_month in [1, 2, 3]:
        return 'q1'
    elif current_month in [4, 5, 6]:
        return 'q2'
    elif current_month in [7, 8, 9]:
        return 'q3'
    else:
        return 'q4'

def validate_environment() -> bool:
    """Validate that all required environment variables are set"""
    try:
        config = get_config()
        logging.info("✅ Configuration validation successful")
        return True
    except Exception as e:
        logging.error("❌ Configuration validation failed: %s", e)
        return False