"""
HDBSCAN Clustering Module
Handles HDBSCAN clustering for the training pipeline
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, Any, List
import hdbscan
import pickle
from pathlib import Path
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class HDBSCANClusterer:
    """
    HDBSCAN clustering processor for reduced embeddings
    """
      def __init__(self, config: Dict[str, Any]):
        """
        Initialize HDBSCAN clusterer with configuration
        
        Args:
            config: Configuration dictionary containing HDBSCAN parameters
        """
        self.config = config
        # Access HDBSCAN config from training.parameters.hdbscan section
        self.hdbscan_config = config.get('training', {}).get('parameters', {}).get('hdbscan', {})
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit HDBSCAN model and predict clusters
        
        Args:
            embeddings: Input embeddings array (n_samples, n_features) - typically UMAP reduced
            
        Returns:
            Tuple of (cluster_labels, metrics)
        """
        try:
            self.logger.info(f"Starting HDBSCAN clustering on {embeddings.shape[0]} samples")
              # Initialize HDBSCAN model with config parameters
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan_config.get('min_cluster_size', 25),
                min_samples=self.hdbscan_config.get('min_samples', 5),
                metric=self.hdbscan_config.get('metric', 'euclidean'),
                cluster_selection_epsilon=self.hdbscan_config.get('cluster_selection_epsilon', 0.0),
                alpha=self.hdbscan_config.get('alpha', 1.0),
                cluster_selection_method=self.hdbscan_config.get('cluster_selection_method', 'eom'),
                core_dist_n_jobs=self.hdbscan_config.get('core_dist_n_jobs', 1)
            )
            
            # Fit and predict
            cluster_labels = self.model.fit_predict(embeddings)
            
            # Calculate metrics
            metrics = self._calculate_metrics(embeddings, cluster_labels)
            
            self.logger.info(f"HDBSCAN clustering completed.")
            self.logger.info(f"Number of clusters: {metrics['n_clusters']}")
            self.logger.info(f"Number of noise points: {metrics['n_noise']}")
            self.logger.info(f"Noise ratio: {metrics['noise_ratio']:.3f}")
            
            return cluster_labels, metrics
            
        except Exception as e:
            self.logger.error(f"HDBSCAN fit_predict failed: {str(e)}")
            raise
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict clusters for new embeddings using fitted HDBSCAN model
        
        Args:
            embeddings: Input embeddings to cluster
            
        Returns:
            Cluster labels
        """
        if self.model is None:
            raise ValueError("HDBSCAN model must be fitted before prediction")
            
        try:
            # Use approximate prediction for new points
            labels, strengths = hdbscan.approximate_predict(self.model, embeddings)
            return labels
        except Exception as e:
            self.logger.error(f"HDBSCAN predict failed: {str(e)}")
            raise
    
    def _calculate_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Calculate clustering quality metrics
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels from HDBSCAN
            
        Returns:
            Dictionary of metrics
        """
        # Basic cluster statistics
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise
        n_noise = np.sum(labels == -1)
        noise_ratio = n_noise / len(labels)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'total_points': len(labels),
            'unique_labels': unique_labels.tolist()
        }
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:  # Exclude noise
                cluster_sizes[int(label)] = np.sum(labels == label)
        metrics['cluster_sizes'] = cluster_sizes
        
        # Calculate quality metrics (only if we have clusters and non-noise points)
        if n_clusters > 1 and n_noise < len(labels):
            try:
                # Filter out noise points for quality metrics
                non_noise_mask = labels != -1
                non_noise_embeddings = embeddings[non_noise_mask]
                non_noise_labels = labels[non_noise_mask]
                
                if len(np.unique(non_noise_labels)) > 1:
                    # Silhouette score
                    silhouette = silhouette_score(non_noise_embeddings, non_noise_labels)
                    metrics['silhouette_score'] = silhouette
                    
                    # Calinski-Harabasz score
                    calinski_harabasz = calinski_harabasz_score(non_noise_embeddings, non_noise_labels)
                    metrics['calinski_harabasz_score'] = calinski_harabasz
                    
                    # Davies-Bouldin score
                    davies_bouldin = davies_bouldin_score(non_noise_embeddings, non_noise_labels)
                    metrics['davies_bouldin_score'] = davies_bouldin
                    
                    self.logger.info(f"Silhouette score: {silhouette:.3f}")
                    self.logger.info(f"Calinski-Harabasz score: {calinski_harabasz:.3f}")
                    self.logger.info(f"Davies-Bouldin score: {davies_bouldin:.3f}")
                    
            except Exception as e:
                self.logger.warning(f"Could not calculate quality metrics: {str(e)}")
                metrics['silhouette_score'] = None
                metrics['calinski_harabasz_score'] = None
                metrics['davies_bouldin_score'] = None
        else:
            self.logger.warning("Not enough clusters or too much noise for quality metrics")
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_score'] = None
            metrics['davies_bouldin_score'] = None
        
        return metrics
    
    def get_cluster_info(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Get detailed information about each cluster
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            
        Returns:
            Dictionary with cluster information
        """
        cluster_info = {}
        
        for label in np.unique(labels):
            if label == -1:  # Skip noise
                continue
                
            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]
            
            cluster_info[int(label)] = {
                'size': np.sum(cluster_mask),
                'centroid': np.mean(cluster_embeddings, axis=0).tolist(),
                'std': np.std(cluster_embeddings, axis=0).tolist(),
                'indices': np.where(cluster_mask)[0].tolist()
            }
        
        return cluster_info
    
    def save_model(self, filepath: str) -> bool:
        """
        Save fitted HDBSCAN model to file
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Success status
        """
        if self.model is None:
            self.logger.error("No HDBSCAN model to save - model must be fitted first")
            return False
            
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
                
            self.logger.info(f"HDBSCAN model saved to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save HDBSCAN model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load HDBSCAN model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
                
            self.logger.info(f"HDBSCAN model loaded from: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load HDBSCAN model: {str(e)}")
            return False
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get HDBSCAN model parameters
        
        Returns:
            Dictionary of model parameters
        """
        if self.model is None:
            return {}
            
        return {
            'min_cluster_size': self.model.min_cluster_size,
            'min_samples': self.model.min_samples,
            'metric': self.model.metric,
            'cluster_selection_epsilon': self.model.cluster_selection_epsilon,
            'alpha': self.model.alpha,
            'cluster_selection_method': self.model.cluster_selection_method
        }
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validate input embeddings for HDBSCAN clustering
        
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
            
        if embeddings.shape[0] < self.hdbscan_config.min_cluster_size:
            self.logger.error(f"Not enough samples ({embeddings.shape[0]}) for min_cluster_size={self.hdbscan_config.min_cluster_size}")
            return False
            
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            self.logger.error("Embeddings contain NaN or infinite values")
            return False
            
        return True
      def validate_clustering_results(self, labels: np.ndarray, metrics: Dict[str, Any]) -> bool:
        """
        Validate clustering results against quality thresholds from config
        
        Args:
            labels: Cluster labels
            metrics: Clustering metrics
            
        Returns:
            True if results meet quality thresholds
        """
        # Get quality thresholds from config
        quality_config = self.config.get('clustering', {}).get('quality', {})
        
        min_clusters = quality_config.get('min_clusters', 3)
        max_noise_ratio = quality_config.get('max_noise_ratio', 0.30)
        min_silhouette_score = quality_config.get('min_silhouette_score', 0.15)
        
        # Check minimum clusters
        if metrics['n_clusters'] < min_clusters:
            self.logger.warning(f"Too few clusters: {metrics['n_clusters']} < {min_clusters}")
            return False
            
        # Check maximum noise ratio
        if metrics['noise_ratio'] > max_noise_ratio:
            self.logger.warning(f"Too much noise: {metrics['noise_ratio']:.3f} > {max_noise_ratio}")
            return False
            
        # Check minimum silhouette score
        if metrics.get('silhouette_score') is not None:
            if metrics['silhouette_score'] < min_silhouette_score:
                self.logger.warning(f"Low silhouette score: {metrics['silhouette_score']:.3f} < {min_silhouette_score}")
                return False
        
        self.logger.info("Clustering results passed quality validation")
        return True