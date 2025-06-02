# data_access/__init__.py
"""
Data access modules for HDBSCAN clustering pipeline.
Provides clients for BigQuery and other data sources.
"""

from .bigquery_client import BigQueryClient

__all__ = ['BigQueryClient']