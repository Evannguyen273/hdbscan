import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class Config:
    """
    Configuration manager that loads settings from YAML and environment variables.
    
    IMPORTANT: This is a template file. Copy to config.py and fill in your actual values.
    The actual config.py file is ignored by git to protect sensitive information.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load environment variables first
        self._load_environment()
        
        # Get configuration path
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = config_path
        
        # Load and process configuration
        self.config = self._load_config(config_path)
        
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
                    logging.warning(f"Failed to parse {env_var} as JSON")
                    return env_value
            
            return env_value
        else:
            return obj
    
    def _validate_config(self):
        """Validate required configuration sections"""
        required_sections = ['bigquery', 'azure', 'clustering']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def bigquery(self):
        """BigQuery configuration"""
        return ConfigSection(self.config.get('bigquery', {}))
    
    @property
    def azure(self):
        """Azure configuration"""
        return ConfigSection(self.config.get('azure', {}))
    
    @property
    def clustering(self):
        """Clustering configuration"""
        return ConfigSection(self.config.get('clustering', {}))
    
    @property
    def tech_centers(self):
        """List of tech centers"""
        return self.config.get('tech_centers', [])

class ConfigSection:
    """Helper class for accessing nested configuration sections"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from configuration section"""
        return self._config.get(key, default)
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access"""
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

# Global configuration instance
_config_instance: Optional[Config] = None

def load_config(config_path: Optional[str] = None) -> Config:
    """Load and return configuration instance"""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance

def get_config() -> Config:
    """Get current configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance

def validate_environment() -> bool:
    """Validate that all required environment variables are set"""
    required_vars = [
        'SERVICE_ACCOUNT_KEY_PATH',
        'TEAM_SERVICES_TABLE',
        'INCIDENT_TABLE', 
        'PROBLEM_TABLE',
        'AZURE_OPENAI_ENDPOINT',
        'OPENAI_API_KEY',
        'AZURE_OPENAI_EMBEDDING_ENDPOINT',
        'BLOB_CONNECTION_STRING'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    return True