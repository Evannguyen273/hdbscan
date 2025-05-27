import logging
import pickle
import json
from azure.storage.blob import BlobServiceClient, BlobClient
from typing import Any, Optional
import os

class BlobStorageClient:
    def __init__(self, config):
        self.config = config
        self.blob_service_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Azure Blob Storage"""
        try:
            if self.config.azure.blob_connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.config.azure.blob_connection_string
                )
                logging.info('Connected to Azure Blob Storage')
            else:
                logging.warning('No blob connection string provided')
        except Exception as e:
            logging.error(f"Failed to connect to Blob Storage: {e}")
    
    def upload_pickle(self, obj: Any, blob_path: str) -> bool:
        """Upload a pickled object to blob storage"""
        if not self.blob_service_client:
            return False
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.config.azure.container_name,
                blob=blob_path
            )
            
            # Serialize object
            pickled_data = pickle.dumps(obj)
            
            # Upload to blob
            blob_client.upload_blob(pickled_data, overwrite=True)
            logging.info(f"Successfully uploaded {blob_path}")
            return True
        except Exception as e:
            logging.error(f"Error uploading {blob_path}: {e}")
            return False
    
    def download_pickle(self, blob_path: str) -> Optional[Any]:
        """Download and unpickle object from blob storage"""
        if not self.blob_service_client:
            return None
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.config.azure.container_name,
                blob=blob_path
            )
            
            # Download blob data
            blob_data = blob_client.download_blob().readall()
            
            # Unpickle object
            obj = pickle.loads(blob_data)
            logging.info(f"Successfully downloaded {blob_path}")
            return obj
        except Exception as e:
            logging.error(f"Error downloading {blob_path}: {e}")
            return None
    
    def upload_json(self, data: dict, blob_path: str) -> bool:
        """Upload JSON data to blob storage"""
        if not self.blob_service_client:
            return False
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.config.azure.container_name,
                blob=blob_path
            )
            
            # Serialize to JSON
            json_data = json.dumps(data, indent=2)
            
            # Upload to blob
            blob_client.upload_blob(json_data, overwrite=True)
            logging.info(f"Successfully uploaded {blob_path}")
            return True
        except Exception as e:
            logging.error(f"Error uploading {blob_path}: {e}")
            return False
    
    def download_json(self, blob_path: str) -> Optional[dict]:
        """Download JSON data from blob storage"""
        if not self.blob_service_client:
            return None
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.config.azure.container_name,
                blob=blob_path
            )
            
            # Download blob data
            blob_data = blob_client.download_blob().readall()
            
            # Parse JSON
            data = json.loads(blob_data.decode('utf-8'))
            logging.info(f"Successfully downloaded {blob_path}")
            return data
        except Exception as e:
            logging.error(f"Error downloading {blob_path}: {e}")
            return None
    
    def list_blobs(self, prefix: str = "") -> list:
        """List blobs with optional prefix filter"""
        if not self.blob_service_client:
            return []
        
        try:
            container_client = self.blob_service_client.get_container_client(
                self.config.azure.container_name
            )
            
            blobs = []
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blobs.append(blob.name)
            
            return blobs
        except Exception as e:
            logging.error(f"Error listing blobs: {e}")
            return []