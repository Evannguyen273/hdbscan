import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

class Config:
    """Configuration manager that handles both config.yaml and enhanced_config.yaml"""
    
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
        
        logging.info(f"Configuration loaded from: {config_path}")
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logging.info(f"Loaded environment variables from {env_file}")
        else:
            logging.warning("No .env file found, using system environment variables")
    
    def _get_default_config_path(self) -> str:
        """Get default config path using config.yaml"""
        config_dir = Path(__file__).parent
        
        # Use main config file
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
                logging.warning(f"Environment variable {env_var} not found")
                return obj
            
            # Handle JSON strings (like SERVICE_ACCOUNT_KEY_PATH)
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
        required_sections = ['bigquery', 'azure', 'clustering']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required configuration section '{section}' not found")
        
        # Validate required environment variables
        required_env_vars = [
            'AZURE_OPENAI_ENDPOINT',
            'OPENAI_API_KEY',
            'BLOB_CONNECTION_STRING'
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logging.error(f"Missing required environment variables: {missing_vars}")
            raise ValueError(f"Missing required environment variables: {missing_vars}")
    
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
    def pipeline(self):
        """Get pipeline configuration"""
        return PipelineConfig(self.config.get('pipeline', {}))
    
    @property
    def tech_centers(self):
        """Get list of tech centers"""
        return self.config.get('tech_centers', [])

class BigQueryConfig:
    """BigQuery configuration wrapper"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @property
    def project_id(self) -> str:
        return self.config.get('project_id', '')
    
    @property
    def service_account_key_path(self):
        return self.config.get('service_account_key_path', '')
    
    @property
    def tables(self) -> Dict[str, str]:
        return self.config.get('tables', {})
    
    def get_table_id(self, table_name: str) -> str:
        """Get full table ID for a table name"""
        return self.tables.get(table_name, '')

class AzureConfig:
    """Azure configuration wrapper"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @property
    def openai(self):
        """Get OpenAI configuration"""
        return self.config.get('openai', {})
    
    @property
    def storage(self):
        """Get storage configuration"""
        return self.config.get('storage', {})
    
    @property
    def blob_storage(self):
        """Get blob storage configuration"""
        return self.config.get('blob_storage', self.storage)

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
    def embedding(self) -> Dict[str, Any]:
        return self.config.get('embedding', {})
    
    @property
    def embedding_weights(self) -> Dict[str, float]:
        return self.embedding.get('weights', {'semantic': 1.0, 'entity': 0.0, 'action': 0.0})
    
    @property
    def domain_grouping(self) -> Dict[str, Any]:
        """Get domain grouping configuration for hybrid approach"""
        return self.config.get('domain_grouping', {
            'enabled': True,
            'max_domains': 20,
            'min_incidents_per_domain': 5,
            'optimization_metric': 'combined',
            'hierarchical_linkage': 'ward'
        })
    
    @property
    def max_domains(self) -> int:
        """Get max domains - backward compatibility"""
        return self.config.get('max_domains', self.domain_grouping.get('max_domains', 20))
    
    @property
    def min_incidents_per_domain(self) -> int:
        """Get min incidents per domain - backward compatibility"""
        return self.config.get('min_incidents_per_domain', self.domain_grouping.get('min_incidents_per_domain', 5))

class PipelineConfig:
    """Pipeline configuration wrapper"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @property
    def save_to_local(self) -> bool:
        return self.config.get('save_to_local', True)
    
    @property
    def result_path(self) -> str:
        return self.config.get('result_path', 'results/')
    
    @property
    def parallel_training(self) -> bool:
        return self.config.get('parallel_training', False)
    
    @property
    def max_workers(self) -> int:
        return self.config.get('max_workers', 4)
    
    @property
    def training_schedule(self) -> Dict[str, Any]:
        return self.config.get('training_schedule', {})
    
    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self.config.get('preprocessing', {})
    
    @property
    def prediction(self) -> Dict[str, Any]:
        return self.config.get('prediction', {})

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
        logging.error(f"❌ Configuration validation failed: {e}")
        return False