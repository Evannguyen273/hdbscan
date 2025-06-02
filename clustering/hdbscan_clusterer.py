# clustering/hdbscan_clusterer.py
# Updated for new config structure and cumulative training approach
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib
import os

from config.config import get_config

class HDBSCANClusterer:
    """
    HDBSCAN clusterer for incident clustering with adaptive parameter tuning.
    Updated for cumulative training approach and versioned model storage.
    """
    
    def __init__(self, config=None):
        """Initialize HDBSCAN clusterer with updated config system"""
        self.config = config if config is not None else get_config()
        
        # Initialize clustering components
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_probabilities = None
        
        # Clustering statistics
        self.clustering_stats = {
            "models_trained": 0,
            "total_points_clustered": 0,
            "training_time": 0,
            "last_cluster_count": 0,
            "last_noise_percentage": 0
        }
        
        # Get clustering configuration
        self.clustering_config = self._get_clustering_config()
        
        logging.info("HDBSCAN clusterer initialized with config")
    
    def _get_clustering_config(self) -> Dict[str, Any]:
        """Get clustering configuration with defaults"""
        hdbscan_config = self.config.clustering.hdbscan
        
        return {
            "min_cluster_size": hdbscan_config.get('min_cluster_size', 5),
            "min_samples": hdbscan_config.get('min_samples', 3),
            "cluster_selection_epsilon": hdbscan_config.get('cluster_selection_epsilon', 0.0),
            "max_cluster_size": hdbscan_config.get('max_cluster_size', None),
            "metric": hdbscan_config.get('metric', 'euclidean'),
            "cluster_selection_method": hdbscan_config.get('cluster_selection_method', 'eom'),
            "algorithm": hdbscan_config.get('algorithm', 'best'),
            "leaf_size": hdbscan_config.get('leaf_size', 40),
            "n_jobs": hdbscan_config.get('n_jobs', -1),
            "adaptive_tuning": hdbscan_config.get('adaptive_tuning', True),
            "auto_optimize": hdbscan_config.get('auto_optimize', True)
        }
    
    def fit_predict(self, embeddings: np.ndarray, 
                   tech_center: str = None,
                   adaptive_params: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Fit HDBSCAN model and predict clusters.
        
        Args:
            embeddings: Input embeddings for clustering
            tech_center: Tech center identifier for logging
            adaptive_params: Whether to use adaptive parameter tuning
            
        Returns:
            Tuple of (cluster_labels, probabilities, clustering_stats)
        """
        training_start = datetime.now()
        
        logging.info("Starting HDBSCAN clustering for %d points%s", 
                    len(embeddings), f" ({tech_center})" if tech_center else "")
        
        try:
            # Validate input
            if len(embeddings) == 0:
                logging.warning("No embeddings provided for clustering")
                return np.array([]), np.array([]), {"status": "failed", "reason": "no_data"}
            
            # Check minimum cluster size requirements
            min_points_required = self.clustering_config["min_cluster_size"] * 2
            if len(embeddings) < min_points_required:
                logging.warning("Insufficient data points (%d < %d) for reliable clustering", 
                              len(embeddings), min_points_required)
                return np.array([-1] * len(embeddings)), np.array([0.0] * len(embeddings)), {
                    "status": "insufficient_data",
                    "points_required": min_points_required,
                    "points_provided": len(embeddings)
                }
            
            # Adaptive parameter tuning if enabled
            if adaptive_params and self.clustering_config["adaptive_tuning"]:
                tuned_params = self._tune_parameters(embeddings)
                clustering_params = {**self.clustering_config, **tuned_params}
            else:
                clustering_params = self.clustering_config.copy()
            
            # Create and fit HDBSCAN clusterer
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=clustering_params["min_cluster_size"],
                min_samples=clustering_params["min_samples"],
                cluster_selection_epsilon=clustering_params["cluster_selection_epsilon"],
                max_cluster_size=clustering_params["max_cluster_size"],
                metric=clustering_params["metric"],
                cluster_selection_method=clustering_params["cluster_selection_method"],
                algorithm=clustering_params["algorithm"],
                leaf_size=clustering_params["leaf_size"],
                n_jobs=clustering_params["n_jobs"]
            )
            
            # Fit and predict
            cluster_labels = self.clusterer.fit_predict(embeddings)
            cluster_probabilities = self.clusterer.probabilities_
            
            # Store results
            self.cluster_labels = cluster_labels
            self.cluster_probabilities = cluster_probabilities
            
            # Calculate clustering statistics
            training_duration = datetime.now() - training_start
            cluster_stats = self._calculate_clustering_statistics(
                cluster_labels, cluster_probabilities, embeddings, training_duration
            )
            
            # Update overall statistics
            self.clustering_stats["models_trained"] += 1
            self.clustering_stats["total_points_clustered"] += len(embeddings)
            self.clustering_stats["training_time"] += training_duration.total_seconds()
            self.clustering_stats["last_cluster_count"] = cluster_stats["n_clusters"]
            self.clustering_stats["last_noise_percentage"] = cluster_stats["noise_percentage"]
            
            logging.info("HDBSCAN clustering completed: %d clusters, %.1f%% noise in %.2f seconds",
                        cluster_stats["n_clusters"], cluster_stats["noise_percentage"], 
                        training_duration.total_seconds())
            
            cluster_stats.update({
                "status": "success",
                "tech_center": tech_center,
                "parameters_used": clustering_params,
                "adaptive_tuning_applied": adaptive_params and self.clustering_config["adaptive_tuning"]
            })
            
            return cluster_labels, cluster_probabilities, cluster_stats
            
        except Exception as e:
            training_duration = datetime.now() - training_start
            logging.error("HDBSCAN clustering failed: %s", str(e))
            
            return np.array([]), np.array([]), {
                "status": "failed",
                "error": str(e),
                "training_duration_seconds": training_duration.total_seconds()
            }
    
    def predict_clusters(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict clusters for new embeddings using fitted model.
        
        Args:
            embeddings: New embeddings to cluster
            
        Returns:
            Tuple of (cluster_labels, probabilities)
        """
        if self.clusterer is None:
            logging.error("No fitted clusterer available for prediction")
            return np.array([-1] * len(embeddings)), np.array([0.0] * len(embeddings))
        
        try:
            # Use approximate prediction for speed
            cluster_labels, probabilities = hdbscan.approximate_predict(
                self.clusterer, embeddings
            )
            
            logging.info("Predicted clusters for %d new points", len(embeddings))
            
            return cluster_labels, probabilities
            
        except Exception as e:
            logging.error("Cluster prediction failed: %s", str(e))
            return np.array([-1] * len(embeddings)), np.array([0.0] * len(embeddings))
    
    def _tune_parameters(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Adaptive parameter tuning based on data characteristics"""
        n_points = len(embeddings)
        
        tuned_params = {}
        
        # Adaptive min_cluster_size based on data size
        base_min_cluster_size = self.clustering_config["min_cluster_size"]
        
        if n_points < 1000:
            # Small dataset: use smaller min_cluster_size
            tuned_params["min_cluster_size"] = max(3, base_min_cluster_size // 2)
        elif n_points > 10000:
            # Large dataset: use larger min_cluster_size
            tuned_params["min_cluster_size"] = min(50, base_min_cluster_size * 2)
        else:
            tuned_params["min_cluster_size"] = base_min_cluster_size
        
        # Adaptive min_samples
        tuned_params["min_samples"] = max(1, tuned_params["min_cluster_size"] // 2)
        
        # Adaptive max_cluster_size for large datasets
        if n_points > 5000:
            tuned_params["max_cluster_size"] = min(n_points // 10, 1000)
        
        logging.info("Adaptive parameter tuning: min_cluster_size=%d, min_samples=%d",
                    tuned_params["min_cluster_size"], tuned_params["min_samples"])
        
        return tuned_params
    
    def _calculate_clustering_statistics(self, cluster_labels: np.ndarray, 
                                       probabilities: np.ndarray,
                                       embeddings: np.ndarray,
                                       training_duration) -> Dict[str, Any]:
        """Calculate detailed clustering statistics"""
        # Basic cluster statistics
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        noise_percentage = (n_noise / len(cluster_labels)) * 100 if len(cluster_labels) > 0 else 0
        
        # Cluster size statistics
        cluster_sizes = []
        if n_clusters > 0:
            for label in unique_labels:
                if label != -1:
                    cluster_size = np.sum(cluster_labels == label)
                    cluster_sizes.append(cluster_size)
        
        # Quality metrics (if we have valid clusters)
        silhouette_avg = None
        calinski_harabasz = None
        
        if n_clusters > 1 and len(embeddings) > n_clusters:
            try:
                # Only calculate for non-noise points
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette_avg = silhouette_score(
                        embeddings[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    calinski_harabasz = calinski_harabasz_score(
                        embeddings[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
            except Exception as e:
                logging.warning("Could not calculate clustering quality metrics: %s", str(e))
        
        # Average cluster probability for non-noise points
        avg_cluster_probability = None
        if len(probabilities) > 0:
            non_noise_probs = probabilities[cluster_labels != -1]
            if len(non_noise_probs) > 0:
                avg_cluster_probability = np.mean(non_noise_probs)
        
        stats = {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_percentage": noise_percentage,
            "total_points": len(cluster_labels),
            "training_duration_seconds": training_duration.total_seconds(),
            "avg_cluster_probability": avg_cluster_probability,
            "silhouette_score": silhouette_avg,
            "calinski_harabasz_score": calinski_harabasz
        }
        
        # Cluster size statistics
        if cluster_sizes:
            stats.update({
                "min_cluster_size": min(cluster_sizes),
                "max_cluster_size": max(cluster_sizes),
                "avg_cluster_size": np.mean(cluster_sizes),
                "median_cluster_size": np.median(cluster_sizes)
            })
        
        return stats
    
    def save_model(self, filepath: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Save the fitted HDBSCAN model to disk.
        
        Args:
            filepath: Path to save the model
            metadata: Additional metadata to save with model
            
        Returns:
            True if successful, False otherwise
        """
        if self.clusterer is None:
            logging.error("No fitted model to save")
            return False
        
        try:
            # Prepare model data
            model_data = {
                "clusterer": self.clusterer,
                "cluster_labels": self.cluster_labels,
                "cluster_probabilities": self.cluster_probabilities,
                "clustering_config": self.clustering_config,
                "clustering_stats": self.clustering_stats,
                "metadata": metadata or {},
                "save_timestamp": datetime.now().isoformat()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(model_data, filepath)
            
            logging.info("HDBSCAN model saved to: %s", filepath)
            return True
            
        except Exception as e:
            logging.error("Failed to save HDBSCAN model: %s", str(e))
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a fitted HDBSCAN model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logging.error("Model file not found: %s", filepath)
                return False
            
            # Load model data
            model_data = joblib.load(filepath)
            
            # Restore model components
            self.clusterer = model_data["clusterer"]
            self.cluster_labels = model_data.get("cluster_labels")
            self.cluster_probabilities = model_data.get("cluster_probabilities")
            self.clustering_config = model_data.get("clustering_config", self.clustering_config)
            self.clustering_stats = model_data.get("clustering_stats", self.clustering_stats)
            
            logging.info("HDBSCAN model loaded from: %s", filepath)
            return True
            
        except Exception as e:
            logging.error("Failed to load HDBSCAN model: %s", str(e))
            return False
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of current clustering results"""
        if self.cluster_labels is None:
            return {"status": "no_model_fitted"}
        
        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        cluster_summary = {
            "n_clusters": n_clusters,
            "n_noise_points": np.sum(self.cluster_labels == -1),
            "total_points": len(self.cluster_labels),
            "cluster_sizes": []
        }
        
        # Get sizes for each cluster
        for label in unique_labels:
            if label != -1:
                cluster_size = np.sum(self.cluster_labels == label)
                cluster_summary["cluster_sizes"].append({
                    "cluster_id": int(label),
                    "size": int(cluster_size)
                })
        
        # Sort by cluster size
        cluster_summary["cluster_sizes"].sort(key=lambda x: x["size"], reverse=True)
        
        return cluster_summary
    
    def get_clustering_statistics(self) -> Dict[str, Any]:
        """Get overall clustering statistics"""
        return self.clustering_stats.copy()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate clustering configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check parameter values
        if self.clustering_config["min_cluster_size"] <= 0:
            validation_results["errors"].append("min_cluster_size must be positive")
            validation_results["valid"] = False
        
        if self.clustering_config["min_samples"] <= 0:
            validation_results["errors"].append("min_samples must be positive")
            validation_results["valid"] = False
        
        if (self.clustering_config["max_cluster_size"] is not None and 
            self.clustering_config["max_cluster_size"] <= self.clustering_config["min_cluster_size"]):
            validation_results["errors"].append("max_cluster_size must be greater than min_cluster_size")
            validation_results["valid"] = False
        
        # Check metric
        valid_metrics = ['euclidean', 'manhattan', 'cosine', 'l1', 'l2']
        if self.clustering_config["metric"] not in valid_metrics:
            validation_results["warnings"].append(f"Unusual metric: {self.clustering_config['metric']}")
        
        # Check cluster selection method
        valid_methods = ['eom', 'leaf']
        if self.clustering_config["cluster_selection_method"] not in valid_methods:
            validation_results["errors"].append(f"Invalid cluster_selection_method: {self.clustering_config['cluster_selection_method']}")
            validation_results["valid"] = False
        
        return validation_results
    
    def reset_statistics(self):
        """Reset clustering statistics"""
        self.clustering_stats = {
            "models_trained": 0,
            "total_points_clustered": 0,
            "training_time": 0,
            "last_cluster_count": 0,
            "last_noise_percentage": 0
        }
        logging.info("HDBSCAN clustering statistics reset")