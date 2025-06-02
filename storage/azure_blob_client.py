# storage/azure_blob_client.py
# Azure Blob Storage client for model storage and versioning
import logging
import json
import pickle
import gzip
import io
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import os

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

from config.config import get_config

class AzureBlobClient:
    """
    Azure Blob Storage client for storing HDBSCAN models and related artifacts.
    Handles versioned model storage with compression and metadata management.
    """
    
    def __init__(self, config=None):
        """Initialize Azure Blob Storage client with updated config system"""
        self.config = config if config is not None else get_config()
        
        # Initialize Azure Blob client
        self._init_blob_client()
        
        # Get storage configuration
        self.storage_config = self._get_storage_config()
        
        # Operation statistics
        self.operation_stats = {
            "blobs_uploaded": 0,
            "blobs_downloaded": 0,
            "containers_created": 0,
            "total_upload_size_mb": 0,
            "total_download_size_mb": 0
        }
        
        logging.info("Azure Blob Storage client initialized for account: %s", 
                    self.storage_config["account_name"])
    
    def _init_blob_client(self):
        """Initialize Azure Blob Storage client"""
        azure_config = self.config.storage.azure_blob
        
        self.account_name = azure_config.get('account_name')
        account_key = azure_config.get('account_key')
        connection_string = azure_config.get('connection_string')
        
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        elif self.account_name and account_key:
            account_url = f"https://{self.account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url, 
                credential=account_key
            )
        else:
            raise ValueError("Azure Blob Storage credentials not properly configured")
    
    def _get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration with defaults"""
        azure_config = self.config.storage.azure_blob
        
        return {
            "account_name": azure_config.get('account_name'),
            "container_name": azure_config.get('container_name', 'hdbscan-models'),
            "enable_compression": azure_config.get('enable_compression', True),
            "compression_level": azure_config.get('compression_level', 6),
            "enable_versioning": azure_config.get('enable_versioning', True),
            "retention_days": azure_config.get('retention_days', 365),
            "metadata_encoding": azure_config.get('metadata_encoding', 'utf-8')
        }
    
    def create_container_if_not_exists(self) -> bool:
        """Create container if it doesn't exist"""
        try:
            container_name = self.storage_config["container_name"]
            
            try:
                container_client = self.blob_service_client.get_container_client(container_name)
                container_client.get_container_properties()
                logging.info("Container already exists: %s", container_name)
                return True
            except ResourceNotFoundError:
                # Create container
                container_client = self.blob_service_client.create_container(container_name)
                
                self.operation_stats["containers_created"] += 1
                logging.info("Created container: %s", container_name)
                return True
                
        except Exception as e:
            logging.error("Failed to create container: %s", str(e))
            return False
    
    def upload_model(self, model_data: Dict[str, Any], 
                    model_version: str, 
                    tech_center: str,
                    model_hash: str) -> bool:
        """
        Upload model data to Azure Blob Storage.
        
        Args:
            model_data: Dictionary containing model and metadata
            model_version: Version identifier
            tech_center: Tech center identifier
            model_hash: Model hash for uniqueness
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create blob name with versioning
            blob_name = self._get_model_blob_name(model_version, tech_center, model_hash)
            
            # Prepare model data with metadata
            upload_data = {
                "model_data": model_data,
                "upload_timestamp": datetime.now().isoformat(),
                "model_version": model_version,
                "tech_center": tech_center,
                "model_hash": model_hash
            }
            
            # Serialize and optionally compress
            serialized_data = pickle.dumps(upload_data)
            
            if self.storage_config["enable_compression"]:
                compressed_data = gzip.compress(
                    serialized_data, 
                    compresslevel=self.storage_config["compression_level"]
                )
                data_to_upload = compressed_data
                content_encoding = "gzip"
            else:
                data_to_upload = serialized_data
                content_encoding = None
            
            # Prepare metadata
            blob_metadata = {
                "model_version": model_version,
                "tech_center": tech_center,
                "model_hash": model_hash,
                "upload_date": datetime.now().strftime("%Y-%m-%d"),
                "compressed": str(self.storage_config["enable_compression"]),
                "size_mb": str(round(len(data_to_upload) / 1024 / 1024, 2))
            }
            
            # Upload to blob storage
            blob_client = self.blob_service_client.get_blob_client(
                container=self.storage_config["container_name"],
                blob=blob_name
            )
            
            blob_client.upload_blob(
                data_to_upload,
                overwrite=True,
                metadata=blob_metadata,
                content_settings={
                    "content_type": "application/octet-stream",
                    "content_encoding": content_encoding
                }
            )
            
            # Update statistics
            size_mb = len(data_to_upload) / 1024 / 1024
            self.operation_stats["blobs_uploaded"] += 1
            self.operation_stats["total_upload_size_mb"] += size_mb
            
            logging.info("Uploaded model %s for %s (%.2f MB)", 
                        model_version, tech_center, size_mb)
            return True
            
        except Exception as e:
            logging.error("Failed to upload model: %s", str(e))
            return False
    
    def download_model(self, model_version: str, 
                      tech_center: str, 
                      model_hash: str = None) -> Optional[Dict[str, Any]]:
        """
        Download model data from Azure Blob Storage.
        
        Args:
            model_version: Version identifier
            tech_center: Tech center identifier
            model_hash: Model hash (optional, will find latest if not provided)
            
        Returns:
            Model data dictionary or None if not found
        """
        try:
            # Find blob name
            if model_hash:
                blob_name = self._get_model_blob_name(model_version, tech_center, model_hash)
            else:
                blob_name = self._find_latest_model_blob(model_version, tech_center)
                
            if not blob_name:
                logging.warning("No model found for version %s, tech center %s", 
                              model_version, tech_center)
                return None
            
            # Download blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.storage_config["container_name"],
                blob=blob_name
            )
            
            download_stream = blob_client.download_blob()
            blob_data = download_stream.readall()
            
            # Get blob properties for metadata
            blob_properties = blob_client.get_blob_properties()
            is_compressed = blob_properties.metadata.get("compressed", "False") == "True"
            
            # Decompress if needed
            if is_compressed:
                decompressed_data = gzip.decompress(blob_data)
                model_data = pickle.loads(decompressed_data)
            else:
                model_data = pickle.loads(blob_data)
            
            # Update statistics
            size_mb = len(blob_data) / 1024 / 1024
            self.operation_stats["blobs_downloaded"] += 1
            self.operation_stats["total_download_size_mb"] += size_mb
            
            logging.info("Downloaded model %s for %s (%.2f MB)", 
                        model_version, tech_center, size_mb)
            
            return model_data
            
        except ResourceNotFoundError:
            logging.warning("Model not found: %s, %s", model_version, tech_center)
            return None
        except Exception as e:
            logging.error("Failed to download model: %s", str(e))
            return None
    
    def list_model_versions(self, tech_center: str = None) -> List[Dict[str, Any]]:
        """
        List available model versions.
        
        Args:
            tech_center: Optional tech center filter
            
        Returns:
            List of model version information
        """
        try:
            container_client = self.blob_service_client.get_container_client(
                self.storage_config["container_name"]
            )
            
            models = []
            blob_prefix = f"models/{tech_center}/" if tech_center else "models/"
            
            for blob in container_client.list_blobs(name_starts_with=blob_prefix):
                blob_metadata = blob.metadata or {}
                
                model_info = {
                    "blob_name": blob.name,
                    "model_version": blob_metadata.get("model_version"),
                    "tech_center": blob_metadata.get("tech_center"),
                    "model_hash": blob_metadata.get("model_hash"),
                    "upload_date": blob_metadata.get("upload_date"),
                    "size_mb": float(blob_metadata.get("size_mb", 0)),
                    "last_modified": blob.last_modified
                }
                models.append(model_info)
            
            # Sort by last modified (newest first)
            models.sort(key=lambda x: x["last_modified"], reverse=True)
            
            logging.info("Found %d model versions%s", 
                        len(models), f" for {tech_center}" if tech_center else "")
            
            return models
            
        except Exception as e:
            logging.error("Failed to list model versions: %s", str(e))
            return []
    
    def delete_model(self, model_version: str, 
                    tech_center: str, 
                    model_hash: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_version: Version identifier
            tech_center: Tech center identifier
            model_hash: Model hash
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob_name = self._get_model_blob_name(model_version, tech_center, model_hash)
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.storage_config["container_name"],
                blob=blob_name
            )
            
            blob_client.delete_blob()
            
            logging.info("Deleted model %s for %s", model_version, tech_center)
            return True
            
        except ResourceNotFoundError:
            logging.warning("Model not found for deletion: %s, %s", model_version, tech_center)
            return False
        except Exception as e:
            logging.error("Failed to delete model: %s", str(e))
            return False
    
    def cleanup_old_models(self, retention_days: int = None) -> int:
        """
        Clean up old model versions based on retention policy.
        
        Args:
            retention_days: Days to retain (uses config default if not provided)
            
        Returns:
            Number of models deleted
        """
        try:
            retention_days = retention_days or self.storage_config["retention_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            container_client = self.blob_service_client.get_container_client(
                self.storage_config["container_name"]
            )
            
            deleted_count = 0
            
            for blob in container_client.list_blobs(name_starts_with="models/"):
                if blob.last_modified and blob.last_modified.replace(tzinfo=None) < cutoff_date:
                    try:
                        blob_client = self.blob_service_client.get_blob_client(
                            container=self.storage_config["container_name"],
                            blob=blob.name
                        )
                        blob_client.delete_blob()
                        deleted_count += 1
                        
                    except Exception as e:
                        logging.error("Failed to delete old blob %s: %s", blob.name, str(e))
            
            logging.info("Cleaned up %d old models (older than %d days)", 
                        deleted_count, retention_days)
            
            return deleted_count
            
        except Exception as e:
            logging.error("Failed to cleanup old models: %s", str(e))
            return 0
    
    def _get_model_blob_name(self, model_version: str, 
                           tech_center: str, 
                           model_hash: str) -> str:
        """Generate blob name for model storage"""
        return f"models/{tech_center}/{model_version}_{model_hash}.pkl"
    
    def _find_latest_model_blob(self, model_version: str, 
                              tech_center: str) -> Optional[str]:
        """Find the latest model blob for a version and tech center"""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.storage_config["container_name"]
            )
            
            blob_prefix = f"models/{tech_center}/{model_version}_"
            latest_blob = None
            latest_modified = None
            
            for blob in container_client.list_blobs(name_starts_with=blob_prefix):
                if latest_modified is None or blob.last_modified > latest_modified:
                    latest_blob = blob.name
                    latest_modified = blob.last_modified
            
            return latest_blob
            
        except Exception as e:
            logging.error("Failed to find latest model blob: %s", str(e))
            return None
    
    def upload_training_artifacts(self, artifacts: Dict[str, Any], 
                                model_version: str, 
                                tech_center: str) -> bool:
        """
        Upload training artifacts (logs, metrics, etc.).
        
        Args:
            artifacts: Dictionary of artifacts to upload
            model_version: Version identifier
            tech_center: Tech center identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob_name = f"artifacts/{tech_center}/{model_version}_artifacts.json"
            
            # Add timestamp
            artifacts["upload_timestamp"] = datetime.now().isoformat()
            
            # Convert to JSON
            json_data = json.dumps(artifacts, indent=2, default=str)
            data_bytes = json_data.encode(self.storage_config["metadata_encoding"])
            
            # Upload
            blob_client = self.blob_service_client.get_blob_client(
                container=self.storage_config["container_name"],
                blob=blob_name
            )
            
            blob_client.upload_blob(
                data_bytes,
                overwrite=True,
                metadata={
                    "model_version": model_version,
                    "tech_center": tech_center,
                    "upload_date": datetime.now().strftime("%Y-%m-%d"),
                    "content_type": "artifacts"
                }
            )
            
            logging.info("Uploaded training artifacts for %s, %s", model_version, tech_center)
            return True
            
        except Exception as e:
            logging.error("Failed to upload training artifacts: %s", str(e))
            return False
    
    def download_training_artifacts(self, model_version: str, 
                                  tech_center: str) -> Optional[Dict[str, Any]]:
        """
        Download training artifacts.
        
        Args:
            model_version: Version identifier
            tech_center: Tech center identifier
            
        Returns:
            Artifacts dictionary or None if not found
        """
        try:
            blob_name = f"artifacts/{tech_center}/{model_version}_artifacts.json"
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.storage_config["container_name"],
                blob=blob_name
            )
            
            download_stream = blob_client.download_blob()
            json_data = download_stream.readall().decode(self.storage_config["metadata_encoding"])
            
            artifacts = json.loads(json_data)
            
            logging.info("Downloaded training artifacts for %s, %s", model_version, tech_center)
            return artifacts
            
        except ResourceNotFoundError:
            logging.warning("Training artifacts not found: %s, %s", model_version, tech_center)
            return None
        except Exception as e:
            logging.error("Failed to download training artifacts: %s", str(e))
            return None
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage operation statistics"""
        return self.operation_stats.copy()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate Azure Blob Storage configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        if not self.storage_config["account_name"]:
            validation_results["errors"].append("Azure Storage account name not configured")
            validation_results["valid"] = False
        
        if not self.storage_config["container_name"]:
            validation_results["errors"].append("Azure Storage container name not configured")
            validation_results["valid"] = False
        
        # Test connection
        try:
            self.blob_service_client.get_account_information()
        except Exception as e:
            validation_results["errors"].append(f"Azure Storage connection failed: {str(e)}")
            validation_results["valid"] = False
        
        return validation_results
    
    def reset_statistics(self):
        """Reset operation statistics"""
        self.operation_stats = {
            "blobs_uploaded": 0,
            "blobs_downloaded": 0,
            "containers_created": 0,
            "total_upload_size_mb": 0,
            "total_download_size_mb": 0
        }
        logging.info("Azure Blob Storage client statistics reset")