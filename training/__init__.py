"""
Training Module for HDBSCAN Clustering Pipeline

This module contains the core training components:
- UMAPProcessor: Dimensionality reduction using UMAP
- HDBSCANClusterer: Clustering using HDBSCAN

Usage:
    from training.umap import UMAPProcessor
    from training.hdbscan_clustering import HDBSCANClusterer
    from pipeline.training_pipeline import TrainingPipeline
"""

from .umap import UMAPProcessor
from .hdbscan_clustering import HDBSCANClusterer

__all__ = [
    'UMAPProcessor',
    'HDBSCANClusterer'
]