# storage/__init__.py
"""
Storage modules for HDBSCAN clustering pipeline.
Provides clients for Azure Blob Storage and other storage systems.
"""

from .azure_blob_client import AzureBlobClient

__all__ = ['AzureBlobClient']