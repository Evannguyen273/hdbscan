"""
UMAP Dimensionality Reduction Module
Handles UMAP fitting and transformation for HDBSCAN clustering pipeline
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, Any
import umap.umap_ as umap
import pickle
from pathlib import Path

class UMAPProcessor:
    """
    UMAP dimensionality reduction processor for embedding data
    """
      def __init__(self, config: Dict[str, Any]):
        """
        Initialize UMAP processor with configuration
        
        Args:
            config: Configuration dictionary containing UMAP parameters
        """
        self.config = config
        # Access UMAP config from training.parameters.umap section
        self.umap_config = config.get('training', {}).get('parameters', {}).get('umap', {})
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def fit_transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit UMAP model and transform embeddings
        
        Args:
            embeddings: Input embeddings array (n_samples, n_features)
            
        Returns:
            Tuple of (reduced_embeddings, metrics)
        """
        try:
            self.logger.info(f"Starting UMAP dimensionality reduction on {embeddings.shape[0]} samples")
              # Initialize UMAP model with config parameters
            self.model = umap.UMAP(
                n_neighbors=self.umap_config.get('n_neighbors', 15),
                n_components=self.umap_config.get('n_components', 2),
                metric=self.umap_config.get('metric', 'cosine'),
                min_dist=self.umap_config.get('min_dist', 0.1),
                spread=self.umap_config.get('spread', 1.0),
                random_state=self.umap_config.get('random_state', 42),
                n_jobs=self.umap_config.get('n_jobs', 1),
                verbose=True
            )
            
            # Fit and transform
            reduced_embeddings = self.model.fit_transform(embeddings)
            
            # Calculate metrics
            metrics = self._calculate_metrics(embeddings, reduced_embeddings)
            
            self.logger.info(f"UMAP reduction completed. Output shape: {reduced_embeddings.shape}")
            self.logger.info(f"Explained variance ratio: {metrics.get('explained_variance_ratio', 'N/A')}")
            
            return reduced_embeddings, metrics
            
        except Exception as e:
            self.logger.error(f"UMAP fit_transform failed: {str(e)}")
            raise
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform new embeddings using fitted UMAP model
        
        Args:
            embeddings: Input embeddings to transform
            
        Returns:
            Reduced embeddings
        """
        if self.model is None:
            raise ValueError("UMAP model must be fitted before transformation")
            
        try:
            return self.model.transform(embeddings)
        except Exception as e:
            self.logger.error(f"UMAP transform failed: {str(e)}")
            raise
    
    def _calculate_metrics(self, original: np.ndarray, reduced: np.ndarray) -> Dict[str, Any]:
        """
        Calculate UMAP quality metrics
        
        Args:
            original: Original high-dimensional embeddings
            reduced: UMAP reduced embeddings
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'original_dimensions': original.shape[1],
            'reduced_dimensions': reduced.shape[1],
            'n_samples': original.shape[0],
            'reduction_ratio': original.shape[1] / reduced.shape[1]
        }
        
        try:
            # Calculate variance explained (approximation)
            original_var = np.var(original, axis=0).sum()
            reduced_var = np.var(reduced, axis=0).sum()
            metrics['explained_variance_ratio'] = reduced_var / original_var
            
        except Exception as e:
            self.logger.warning(f"Could not calculate explained variance: {str(e)}")
            metrics['explained_variance_ratio'] = None
        
        return metrics
    
    def save_model(self, filepath: str) -> bool:
        """
        Save fitted UMAP model to file
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Success status
        """
        if self.model is None:
            self.logger.error("No UMAP model to save - model must be fitted first")
            return False
            
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
                
            self.logger.info(f"UMAP model saved to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save UMAP model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load UMAP model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
                
            self.logger.info(f"UMAP model loaded from: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load UMAP model: {str(e)}")
            return False
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get UMAP model parameters
        
        Returns:
            Dictionary of model parameters
        """
        if self.model is None:
            return {}
            
        return {
            'n_neighbors': self.model.n_neighbors,
            'n_components': self.model.n_components,
            'metric': self.model.metric,
            'min_dist': self.model.min_dist,
            'spread': self.model.spread,
            'random_state': self.model.random_state
        }
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validate input embeddings for UMAP processing
        
        Args:
            embeddings: Input embeddings array
            
        Returns:
            True if valid, False otherwise
        """
        if embeddings is None or embeddings.size == 0:
            self.logger.error("Embeddings array is empty")
            return False
            
        if len(embeddings.shape) != 2:
            self.logger.error(f"Embeddings must be 2D array, got shape: {embeddings.shape}")
            return False
            
        if embeddings.shape[0] < self.umap_config.n_neighbors:
            self.logger.error(f"Not enough samples ({embeddings.shape[0]}) for n_neighbors={self.umap_config.n_neighbors}")
            return False
            
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            self.logger.error("Embeddings contain NaN or infinite values")
            return False
            
        return True