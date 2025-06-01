import logging
import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError
import pandas as pd

class BlobStorageClient:
    def __init__(self, config):
        self.config = config
        
        # Get connection string from config (already substituted from environment)
        blob_config = config.get('blob_storage', {})
        self.connection_string = blob_config.get('connection_string', '')
        
        # Extract account URL from connection string for BlobServiceClient
        if self.connection_string.startswith('https://'):
            # Direct URL format
            self.account_url = self.connection_string
            self.client = BlobServiceClient(account_url=self.account_url)
        else:
            # Standard connection string format
            self.client = BlobServiceClient.from_connection_string(self.connection_string)
        
        self.container_name = blob_config.get('container_name', 'prediction-artifacts')
        
        # Create container if it doesn't exist
        try:
            self.client.create_container(self.container_name)
            logging.info(f"Created container: {self.container_name}")
        except Exception:
            logging.info(f"Container {self.container_name} already exists")
        
        logging.info(f"BlobStorageClient initialized for container: {self.container_name}")
    
    def _get_blob_path(self, tech_center: str, year: int, quarter: str, artifact_type: str, filename: str) -> str:
        """Generate standardized blob path"""
        # Clean tech center name for path
        clean_tech_center = tech_center.replace(" ", "_").replace("&", "and")
        
        path_templates = {
            'model': f"models/{clean_tech_center}/{year}/{quarter}/{filename}",
            'prediction': f"predictions/{clean_tech_center}/{year}/{quarter}/{filename}",
            'monitoring': f"monitoring/{clean_tech_center}/{filename}",
            'preprocessing': f"preprocessing/{clean_tech_center}/{year}/{quarter}/{filename}"
        }
        
        return path_templates.get(artifact_type, f"{artifact_type}/{clean_tech_center}/{year}/{quarter}/{filename}")
    
    def save_model_artifacts(self, tech_center: str, year: int, quarter: str, artifacts: Dict[str, Any]) -> bool:
        """Save all model artifacts for a tech center and quarter"""
        try:
            success_count = 0
            total_artifacts = len(artifacts)
            
            for artifact_name, artifact_data in artifacts.items():
                blob_path = self._get_blob_path(tech_center, year, quarter, 'model', artifact_name)
                
                if self._save_artifact(blob_path, artifact_data, artifact_name):
                    success_count += 1
                else:
                    logging.error(f"Failed to save {artifact_name} for {tech_center}")
            
            if success_count == total_artifacts:
                logging.info(f"Successfully saved all {total_artifacts} artifacts for {tech_center} {year}-{quarter}")
                return True
            else:
                logging.warning(f"Saved {success_count}/{total_artifacts} artifacts for {tech_center} {year}-{quarter}")
                return False
                
        except Exception as e:
            logging.error(f"Error saving model artifacts for {tech_center}: {e}")
            return False
    
    def _save_artifact(self, blob_path: str, data: Any, artifact_name: str) -> bool:
        """Save individual artifact to blob storage"""
        try:
            blob_client = self.client.get_blob_client(
                container=self.container_name, 
                blob=blob_path
            )
            
            # Handle different data types
            if artifact_name.endswith('.pkl'):
                # Pickle serialization for models
                serialized_data = pickle.dumps(data)
                blob_client.upload_blob(serialized_data, overwrite=True)
                
            elif artifact_name.endswith('.json'):
                # JSON serialization for metadata
                json_data = json.dumps(data, indent=2, default=str)
                blob_client.upload_blob(json_data, overwrite=True)
                
            elif isinstance(data, pd.DataFrame):
                # CSV for DataFrames
                csv_data = data.to_csv(index=False)
                blob_client.upload_blob(csv_data, overwrite=True)
                
            else:
                # String or bytes data
                if isinstance(data, str):
                    blob_client.upload_blob(data, overwrite=True)
                else:
                    blob_client.upload_blob(data, overwrite=True)
            
            logging.info(f"Saved {artifact_name} to {blob_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save {artifact_name}: {e}")
            return False
    
    def load_model_artifacts(self, tech_center: str, year: int, quarter: str) -> Optional[Dict[str, Any]]:
        """Load all model artifacts for a tech center and quarter"""
        try:
            artifacts = {}
            
            # List all blobs in the model directory
            clean_tech_center = tech_center.replace(" ", "_").replace("&", "and")
            prefix = f"models/{clean_tech_center}/{year}/{quarter}/"
            
            blob_list = self.client.get_container_client(self.container_name).list_blobs(name_starts_with=prefix)
            
            for blob in blob_list:
                artifact_name = blob.name.split('/')[-1]  # Get filename
                artifact_data = self._load_artifact(blob.name, artifact_name)
                
                if artifact_data is not None:
                    artifacts[artifact_name] = artifact_data
            
            if artifacts:
                logging.info(f"Loaded {len(artifacts)} artifacts for {tech_center} {year}-{quarter}")
                return artifacts
            else:
                logging.warning(f"No artifacts found for {tech_center} {year}-{quarter}")
                return None
                
        except Exception as e:
            logging.error(f"Error loading model artifacts for {tech_center}: {e}")
            return None
    
    def _load_artifact(self, blob_path: str, artifact_name: str) -> Optional[Any]:
        """Load individual artifact from blob storage"""
        try:
            blob_client = self.client.get_blob_client(
                container=self.container_name, 
                blob=blob_path
            )
            
            blob_data = blob_client.download_blob().readall()
            
            # Handle different data types
            if artifact_name.endswith('.pkl'):
                return pickle.loads(blob_data)
                
            elif artifact_name.endswith('.json'):
                return json.loads(blob_data.decode('utf-8'))
                
            elif artifact_name.endswith('.csv'):
                import io
                return pd.read_csv(io.BytesIO(blob_data))
                
            else:
                # Return as string
                return blob_data.decode('utf-8')
                
        except ResourceNotFoundError:
            logging.warning(f"Artifact not found: {blob_path}")
            return None
        except Exception as e:
            logging.error(f"Failed to load {artifact_name}: {e}")
            return None
    
    def save_predictions(self, tech_center: str, year: int, quarter: str, predictions_df: pd.DataFrame, 
                        prediction_type: str = "batch") -> bool:
        """Save prediction results"""
        try:
            filename = f"{prediction_type}_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            blob_path = self._get_blob_path(tech_center, year, quarter, 'prediction', filename)
            
            return self._save_artifact(blob_path, predictions_df, filename)
            
        except Exception as e:
            logging.error(f"Error saving predictions for {tech_center}: {e}")
            return False
    
    def list_available_models(self, tech_center: str = None) -> Dict[str, Dict]:
        """List all available models, optionally filtered by tech center"""
        try:
            models = {}
            
            if tech_center:
                clean_tech_center = tech_center.replace(" ", "_").replace("&", "and")
                prefix = f"models/{clean_tech_center}/"
            else:
                prefix = "models/"
            
            blob_list = self.client.get_container_client(self.container_name).list_blobs(name_starts_with=prefix)
            
            for blob in blob_list:
                path_parts = blob.name.split('/')
                if len(path_parts) >= 4:  # models/tech_center/year/quarter/file
                    tc = path_parts[1].replace("_", " ").replace("and", "&")
                    year = path_parts[2]
                    quarter = path_parts[3]
                    
                    model_key = f"{tc}_{year}_{quarter}"
                    
                    if model_key not in models:
                        models[model_key] = {
                            'tech_center': tc,
                            'year': int(year),
                            'quarter': quarter,
                            'artifacts': [],
                            'last_modified': blob.last_modified
                        }
                    
                    models[model_key]['artifacts'].append(path_parts[4])
            
            return models
            
        except Exception as e:
            logging.error(f"Error listing models: {e}")
            return {}
    
    def get_latest_model_path(self, tech_center: str) -> Optional[Dict[str, Any]]:
        """Get the path to the latest model for a tech center"""
        try:
            models = self.list_available_models(tech_center)
            
            if not models:
                return None
            
            # Find the latest model by year and quarter
            latest_model = max(models.values(), key=lambda x: (x['year'], x['quarter']))
            
            return {
                'tech_center': latest_model['tech_center'],
                'year': latest_model['year'],
                'quarter': latest_model['quarter'],
                'path_prefix': f"models/{tech_center.replace(' ', '_').replace('&', 'and')}/{latest_model['year']}/{latest_model['quarter']}/",
                'artifacts': latest_model['artifacts']
            }
            
        except Exception as e:
            logging.error(f"Error finding latest model for {tech_center}: {e}")
            return None
    
    def delete_old_models(self, tech_center: str, keep_last_n: int = 2) -> bool:
        """Delete old models, keeping only the last N quarters"""
        try:
            models = self.list_available_models(tech_center)
            
            if len(models) <= keep_last_n:
                logging.info(f"Only {len(models)} models found for {tech_center}, nothing to delete")
                return True
            
            # Sort models by year and quarter
            sorted_models = sorted(models.values(), key=lambda x: (x['year'], x['quarter']), reverse=True)
            models_to_delete = sorted_models[keep_last_n:]
            
            deleted_count = 0
            for model in models_to_delete:
                clean_tech_center = tech_center.replace(" ", "_").replace("&", "and")
                prefix = f"models/{clean_tech_center}/{model['year']}/{model['quarter']}/"
                
                # Delete all blobs with this prefix
                blob_list = self.client.get_container_client(self.container_name).list_blobs(name_starts_with=prefix)
                
                for blob in blob_list:
                    try:
                        self.client.delete_blob(container=self.container_name, blob=blob.name)
                        deleted_count += 1
                    except Exception as e:
                        logging.error(f"Failed to delete blob {blob.name}: {e}")
            
            logging.info(f"Deleted {deleted_count} old model artifacts for {tech_center}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting old models for {tech_center}: {e}")
            return False