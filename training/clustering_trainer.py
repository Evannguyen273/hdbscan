# filepath: c:\Users\Bachn\OneDrive\Desktop\my_projects\hdbscan\training\clustering_trainer.py
import logging
import numpy as np
import pandas as pd
import joblib
import os
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime
import warnings
from pathlib import Path

# Clustering and evaluation imports
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ClusteringTrainer:
    """Comprehensive clustering trainer with extensive failure handling"""
    
    def __init__(self, config):
        self.config = config
        
        # Track training failures
        self.training_failures: List[str] = []
        self.training_warnings: List[str] = []
        
        # Training results storage
        self.trained_model = None
        self.cluster_labels = None
        self.training_stats = {}
          # Configurable minimum requirements for training
        self.min_samples_for_training = config.get('min_samples_for_training', 20)  # Reduced from 50
        self.min_embedding_dimensions = config.get('min_embedding_dimensions', 50)  # Reduced from 100
        self.max_memory_usage_gb = config.get('max_memory_usage_gb', 8.0)
        
        # Warning thresholds (show warnings but allow training)
        self.recommended_min_samples = 50
        self.recommended_min_dimensions = 100
          def validate_training_data(self, embedding_matrix: np.ndarray, 
                             valid_indices: pd.Index) -> Tuple[bool, List[str]]:
        """
        Validate that data is suitable for clustering training.
        
        Args:
            embedding_matrix: Matrix of embeddings for clustering
            valid_indices: Indices of valid embeddings
            
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        validation_errors = []
      def validate_training_data(self, embedding_matrix: np.ndarray, 
                             valid_indices: pd.Index) -> Tuple[bool, List[str]]:
        """
        Validate that data is suitable for clustering training.
        
        Args:
            embedding_matrix: Matrix of embeddings for clustering
            valid_indices: Indices of valid embeddings
            
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        validation_errors = []
        
        # Check basic requirements
        if len(embedding_matrix) == 0:
            validation_errors.append("No embeddings available for training")
            return False, validation_errors
        
        # Hard minimum for clustering to be meaningful
        if len(embedding_matrix) < self.min_samples_for_training:
            validation_errors.append(f"Insufficient samples: {len(embedding_matrix)} < {self.min_samples_for_training}")
        
        # Soft warnings for suboptimal conditions
        if len(embedding_matrix) < self.recommended_min_samples:
            warning_msg = f"Warning: Small dataset ({len(embedding_matrix)} samples). Recommend {self.recommended_min_samples}+ for stable clustering"
            self.training_warnings.append(warning_msg)
            logging.warning("⚠️ %s", warning_msg)
        
        # Check embedding dimensions
        if embedding_matrix.shape[1] < self.min_embedding_dimensions:
            validation_errors.append(f"Embedding dimensions too low: {embedding_matrix.shape[1]} < {self.min_embedding_dimensions}")
        
        if embedding_matrix.shape[1] < self.recommended_min_dimensions:
            warning_msg = f"Warning: Low-dimensional embeddings ({embedding_matrix.shape[1]}D). Recommend {self.recommended_min_dimensions}+ for better clustering"
            self.training_warnings.append(warning_msg)
            logging.warning("⚠️ %s", warning_msg)
        
        # Check for invalid values
        if np.isnan(embedding_matrix).any():
            validation_errors.append("Embedding matrix contains NaN values")
        
        if np.isinf(embedding_matrix).any():
            validation_errors.append("Embedding matrix contains infinite values")
        
        # Check memory requirements
        estimated_memory_gb = (embedding_matrix.nbytes * 3) / (1024**3)  # Conservative estimate
        if estimated_memory_gb > self.max_memory_usage_gb:
            validation_errors.append(f"Estimated memory usage too high: {estimated_memory_gb:.1f}GB > {self.max_memory_usage_gb}GB")
          # Check data variance (warn about low diversity but don't stop training)
        if np.var(embedding_matrix) < 1e-10:
            warning_msg = "Warning: Very low embedding variance detected - many similar incidents. This is normal for repetitive issues but may limit clustering effectiveness"
            self.training_warnings.append(warning_msg)
            logging.warning("⚠️ %s", warning_msg)
        
        # Check for duplicate rows (more lenient threshold)
        unique_rows = np.unique(embedding_matrix, axis=0)
        duplicate_ratio = 1 - (len(unique_rows) / len(embedding_matrix))
        if duplicate_ratio > 0.5:  # More than 50% duplicates
            validation_errors.append(f"Too many duplicate embeddings: {duplicate_ratio:.1%} of embeddings are identical")
        elif duplicate_ratio > 0.2:  # Warning for moderate duplicates
            warning_msg = f"Warning: Moderate duplicate embeddings detected ({duplicate_ratio:.1%})"
            self.training_warnings.append(warning_msg)
            logging.warning("⚠️ %s", warning_msg)
        
        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors
      def validate_hdbscan_parameters(self, embedding_matrix: np.ndarray, **hdbscan_params) -> Tuple[bool, List[str]]:
        """
        Validate HDBSCAN parameters before training, considering dataset size.
        
        Args:
            embedding_matrix: Embedding matrix to be clustered
            **hdbscan_params: HDBSCAN parameters to validate
            
        Returns:
            Tuple of (are_valid, validation_errors)
        """
        validation_errors = []
        
        min_cluster_size = hdbscan_params.get('min_cluster_size', 15)
        min_samples = hdbscan_params.get('min_samples', None)
        cluster_selection_epsilon = hdbscan_params.get('cluster_selection_epsilon', 0.0)
        
        dataset_size = len(embedding_matrix)
        
        # Validate min_cluster_size
        if not isinstance(min_cluster_size, int) or min_cluster_size < 2:
            validation_errors.append("min_cluster_size must be an integer >= 2")
        
        # Check if min_cluster_size is reasonable for dataset size
        if min_cluster_size > dataset_size // 3:
            validation_errors.append(f"min_cluster_size ({min_cluster_size}) too large for dataset size ({dataset_size}). Should be < {dataset_size // 3}")
        
        # Warn if min_cluster_size might be too large for small datasets
        if dataset_size < 100 and min_cluster_size > dataset_size // 5:
            warning_msg = f"Warning: min_cluster_size ({min_cluster_size}) may be too large for small dataset ({dataset_size} samples). Consider reducing to {dataset_size // 5} or less"
            self.training_warnings.append(warning_msg)
            logging.warning("⚠️ %s", warning_msg)
        
        # Validate min_samples
        if min_samples is not None:
            if not isinstance(min_samples, int) or min_samples < 1:
                validation_errors.append("min_samples must be an integer >= 1")
            
            if min_samples > min_cluster_size:
                validation_errors.append("min_samples should not be larger than min_cluster_size")
        
        # Validate cluster_selection_epsilon
        if not isinstance(cluster_selection_epsilon, (int, float)) or cluster_selection_epsilon < 0:
            validation_errors.append("cluster_selection_epsilon must be a non-negative number")
        
        # Validate metric
        metric = hdbscan_params.get('metric', 'euclidean')
        valid_metrics = ['euclidean', 'manhattan', 'cosine', 'hamming']
        if metric not in valid_metrics:
            validation_errors.append(f"metric must be one of {valid_metrics}")
        
        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors
    
    def preprocess_embeddings_for_training(self, embedding_matrix: np.ndarray, 
                                         apply_scaling: bool = True,
                                         apply_pca: bool = False,
                                         pca_components: int = 50) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess embeddings before training with failure handling.
        
        Args:
            embedding_matrix: Input embeddings
            apply_scaling: Whether to apply standardization
            apply_pca: Whether to apply PCA dimensionality reduction
            pca_components: Number of PCA components
            
        Returns:
            Tuple of (processed_embeddings, preprocessing_info)
        """
        preprocessing_info = {
            "original_shape": embedding_matrix.shape,
            "applied_scaling": False,
            "applied_pca": False,
            "scaling_errors": [],
            "pca_errors": []
        }
        
        processed_embeddings = embedding_matrix.copy()
        
        try:
            # Apply scaling if requested
            if apply_scaling:
                logging.info("Applying standardization to embeddings...")
                scaler = StandardScaler()
                processed_embeddings = scaler.fit_transform(processed_embeddings)
                preprocessing_info["applied_scaling"] = True
                preprocessing_info["scaler"] = scaler
                logging.info("✓ Standardization applied successfully")
                
        except Exception as e:
            error_msg = f"Scaling failed: {str(e)}"
            preprocessing_info["scaling_errors"].append(error_msg)
            self.training_warnings.append(error_msg)
            logging.warning("✗ Scaling failed: %s", error_msg)
        
        try:
            # Apply PCA if requested and beneficial
            if apply_pca and processed_embeddings.shape[1] > pca_components:
                logging.info("Applying PCA dimensionality reduction...")
                pca = PCA(n_components=pca_components, random_state=42)
                processed_embeddings = pca.fit_transform(processed_embeddings)
                
                explained_variance = np.sum(pca.explained_variance_ratio_)
                preprocessing_info["applied_pca"] = True
                preprocessing_info["pca"] = pca
                preprocessing_info["explained_variance_ratio"] = explained_variance
                preprocessing_info["final_shape"] = processed_embeddings.shape
                
                logging.info("✓ PCA applied: %d → %d dimensions, %.1f%% variance retained", 
                           embedding_matrix.shape[1], pca_components, explained_variance * 100)
                
        except Exception as e:
            error_msg = f"PCA failed: {str(e)}"
            preprocessing_info["pca_errors"].append(error_msg)
            self.training_warnings.append(error_msg)
            logging.warning("✗ PCA failed: %s", error_msg)
        
        return processed_embeddings, preprocessing_info
    
    def train_hdbscan_with_fallbacks(self, embedding_matrix: np.ndarray,
                                   primary_params: Dict,
                                   fallback_params_list: List[Dict]) -> Tuple[Optional[hdbscan.HDBSCAN], Dict]:
        """
        Train HDBSCAN with fallback parameter sets if primary fails.
        
        Args:
            embedding_matrix: Preprocessed embeddings
            primary_params: Primary HDBSCAN parameters
            fallback_params_list: List of fallback parameter sets
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        training_info = {
            "primary_params": primary_params,
            "fallback_params": fallback_params_list,
            "successful_params": None,
            "training_errors": [],
            "used_fallback": False
        }
        
        # Try primary parameters first
        logging.info("Training HDBSCAN with primary parameters...")
        logging.info("Parameters: %s", primary_params)
        
        try:
            model = hdbscan.HDBSCAN(**primary_params)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cluster_labels = model.fit_predict(embedding_matrix)
            
            # Validate results
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            if n_clusters == 0:
                raise ValueError("No clusters found - all points classified as noise")
            
            if n_noise > len(cluster_labels) * 0.9:
                raise ValueError(f"Too much noise: {n_noise}/{len(cluster_labels)} points classified as noise")
            
            training_info["successful_params"] = primary_params
            logging.info("✓ Primary parameters successful: %d clusters, %d noise points", n_clusters, n_noise)
            return model, training_info
            
        except Exception as e:
            error_msg = f"Primary parameters failed: {str(e)}"
            training_info["training_errors"].append(error_msg)
            self.training_failures.append(error_msg)
            logging.warning("✗ Primary parameters failed: %s", error_msg)
        
        # Try fallback parameters
        for i, fallback_params in enumerate(fallback_params_list):
            logging.info("Trying fallback parameters %d/%d...", i+1, len(fallback_params_list))
            logging.info("Parameters: %s", fallback_params)
            
            try:
                model = hdbscan.HDBSCAN(**fallback_params)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cluster_labels = model.fit_predict(embedding_matrix)
                
                # Validate results
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                
                if n_clusters == 0:
                    raise ValueError("No clusters found - all points classified as noise")
                
                if n_noise > len(cluster_labels) * 0.9:
                    raise ValueError(f"Too much noise: {n_noise}/{len(cluster_labels)} points classified as noise")
                
                training_info["successful_params"] = fallback_params
                training_info["used_fallback"] = True
                training_info["fallback_index"] = i
                logging.info("✓ Fallback parameters %d successful: %d clusters, %d noise points", 
                           i+1, n_clusters, n_noise)
                return model, training_info
                
            except Exception as e:
                error_msg = f"Fallback parameters {i+1} failed: {str(e)}"
                training_info["training_errors"].append(error_msg)
                logging.warning("✗ Fallback parameters %d failed: %s", i+1, error_msg)
        
        # All parameters failed
        self.training_failures.append("All HDBSCAN parameter sets failed")
        return None, training_info
    
    def calculate_clustering_metrics(self, embedding_matrix: np.ndarray, 
                                   cluster_labels: np.ndarray) -> Tuple[Dict, List[str]]:
        """
        Calculate clustering quality metrics with error handling.
        
        Args:
            embedding_matrix: Embeddings used for clustering
            cluster_labels: Cluster assignments
            
        Returns:
            Tuple of (metrics_dict, metric_errors)
        """
        metrics = {}
        metric_errors = []
        
        # Basic cluster statistics
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        metrics["n_clusters"] = n_clusters
        metrics["n_noise_points"] = n_noise
        metrics["noise_ratio"] = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
        
        # Cluster size statistics
        if n_clusters > 0:
            cluster_sizes = [list(cluster_labels).count(label) for label in unique_labels if label != -1]
            metrics["min_cluster_size"] = min(cluster_sizes)
            metrics["max_cluster_size"] = max(cluster_sizes)
            metrics["mean_cluster_size"] = np.mean(cluster_sizes)
            metrics["std_cluster_size"] = np.std(cluster_sizes)
        
        # Advanced metrics (only if we have meaningful clusters)
        if n_clusters > 1 and n_noise < len(cluster_labels):
            # Silhouette Score
            try:
                # Only calculate for non-noise points
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette_avg = silhouette_score(
                        embedding_matrix[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    metrics["silhouette_score"] = silhouette_avg
                    logging.info("✓ Silhouette score: %.3f", silhouette_avg)
            except Exception as e:
                metric_errors.append(f"Silhouette score calculation failed: {str(e)}")
                logging.warning("✗ Silhouette score calculation failed: %s", str(e))
            
            # Calinski-Harabasz Index
            try:
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 1:
                    ch_score = calinski_harabasz_score(
                        embedding_matrix[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    metrics["calinski_harabasz_score"] = ch_score
                    logging.info("✓ Calinski-Harabasz score: %.3f", ch_score)
            except Exception as e:
                metric_errors.append(f"Calinski-Harabasz score calculation failed: {str(e)}")
                logging.warning("✗ Calinski-Harabasz score calculation failed: %s", str(e))
            
            # Davies-Bouldin Index
            try:
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 1:
                    db_score = davies_bouldin_score(
                        embedding_matrix[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    metrics["davies_bouldin_score"] = db_score
                    logging.info("✓ Davies-Bouldin score: %.3f", db_score)
            except Exception as e:
                metric_errors.append(f"Davies-Bouldin score calculation failed: {str(e)}")
                logging.warning("✗ Davies-Bouldin score calculation failed: %s", str(e))
        
        return metrics, metric_errors
      def validate_clustering_quality(self, metrics: Dict, dataset_size: int) -> Tuple[bool, List[str]]:
        """
        Validate if clustering results meet quality thresholds, adjusted for dataset size.
        
        Args:
            metrics: Clustering metrics dictionary
            dataset_size: Size of the original dataset
            
        Returns:
            Tuple of (is_acceptable, quality_issues)
        """
        quality_issues = []
        
        # Check basic requirements
        if metrics.get("n_clusters", 0) < 2:
            quality_issues.append("Too few clusters found (minimum 2 required)")
        
        # Adjust noise tolerance based on dataset size
        if dataset_size < 50:
            max_noise_ratio = 0.9  # Very lenient for small datasets
        elif dataset_size < 100:
            max_noise_ratio = 0.85  # Lenient for small-medium datasets
        else:
            max_noise_ratio = 0.8  # Standard threshold for larger datasets
        
        if metrics.get("noise_ratio", 1.0) > max_noise_ratio:
            quality_issues.append(f"Too much noise: {metrics['noise_ratio']:.1%} of points (max {max_noise_ratio:.0%} for dataset size {dataset_size})")
        
        # Adjust minimum cluster size based on dataset size
        min_required_cluster_size = max(2, dataset_size // 50)  # At least 2, but scale with dataset
        
        if metrics.get("min_cluster_size", 0) < min_required_cluster_size:
            quality_issues.append(f"Clusters too small (minimum {min_required_cluster_size} points per cluster for dataset size {dataset_size})")
        
        # Check silhouette score if available (more lenient thresholds for small datasets)
        silhouette_score = metrics.get("silhouette_score")
        if silhouette_score is not None:
            if dataset_size < 50:
                min_silhouette = 0.1  # Very lenient for small datasets
            elif dataset_size < 100:
                min_silhouette = 0.15  # Lenient for small-medium datasets
            else:
                min_silhouette = 0.2  # Standard threshold
            
            if silhouette_score < min_silhouette:
                quality_issues.append(f"Poor cluster separation (silhouette score: {silhouette_score:.3f} < {min_silhouette:.1f})")
        
        # Check for extremely unbalanced clusters (more lenient for small datasets)
        max_cluster_size = metrics.get("max_cluster_size", 0)
        mean_cluster_size = metrics.get("mean_cluster_size", 0)
        
        if mean_cluster_size > 0:
            if dataset_size < 50:
                max_imbalance_ratio = 20  # Very lenient
            elif dataset_size < 100:
                max_imbalance_ratio = 15  # Lenient
            else:
                max_imbalance_ratio = 10  # Standard
            
            if max_cluster_size > mean_cluster_size * max_imbalance_ratio:
                quality_issues.append(f"Severely unbalanced cluster sizes detected (largest: {max_cluster_size}, mean: {mean_cluster_size:.1f})")
        
        is_acceptable = len(quality_issues) == 0
        return is_acceptable, quality_issues
    
    def save_training_results(self, model: hdbscan.HDBSCAN, 
                            embedding_matrix: np.ndarray,
                            cluster_labels: np.ndarray,
                            valid_indices: pd.Index,
                            training_stats: Dict,
                            output_dir: str = "models") -> Tuple[bool, List[str]]:
        """
        Save training results with error handling.
        
        Args:
            model: Trained HDBSCAN model
            embedding_matrix: Embeddings used for training
            cluster_labels: Cluster assignments
            valid_indices: Indices of valid data points
            training_stats: Training statistics
            output_dir: Directory to save results
            
        Returns:
            Tuple of (save_successful, save_errors)
        """
        save_errors = []
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = os.path.join(output_dir, f"hdbscan_model_{timestamp}.joblib")
            joblib.dump(model, model_path)
            logging.info("✓ Model saved to: %s", model_path)
            
            # Save cluster labels
            labels_path = os.path.join(output_dir, f"cluster_labels_{timestamp}.npy")
            np.save(labels_path, cluster_labels)
            logging.info("✓ Cluster labels saved to: %s", labels_path)
            
            # Save training statistics
            stats_path = os.path.join(output_dir, f"training_stats_{timestamp}.json")
            import json
            with open(stats_path, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_stats = {}
                for k, v in training_stats.items():
                    if isinstance(v, np.integer):
                        json_stats[k] = int(v)
                    elif isinstance(v, np.floating):
                        json_stats[k] = float(v)
                    else:
                        json_stats[k] = v
                json.dump(json_stats, f, indent=2)
            logging.info("✓ Training statistics saved to: %s", stats_path)
            
            # Save data mapping
            mapping_path = os.path.join(output_dir, f"data_mapping_{timestamp}.npz")
            np.savez_compressed(
                mapping_path,
                embeddings=embedding_matrix,
                indices=valid_indices.values,
                cluster_labels=cluster_labels,
                timestamp=timestamp
            )
            logging.info("✓ Data mapping saved to: %s", mapping_path)
            
            return True, save_errors
            
        except Exception as e:
            save_errors.append(f"Failed to save training results: {str(e)}")
            logging.error("✗ Failed to save training results: %s", str(e))
            return False, save_errors
    
    def run_complete_training(self, embedding_matrix: np.ndarray, 
                            valid_indices: pd.Index,
                            hdbscan_params: Dict,
                            fallback_params_list: List[Dict] = None,
                            preprocessing_config: Dict = None,
                            output_dir: str = "models") -> Tuple[bool, Dict]:
        """
        Run complete clustering training pipeline with comprehensive error handling.
        
        Args:
            embedding_matrix: Embeddings for clustering
            valid_indices: Valid data indices
            hdbscan_params: Primary HDBSCAN parameters
            fallback_params_list: Fallback parameter sets
            preprocessing_config: Preprocessing configuration
            output_dir: Output directory for results
            
        Returns:
            Tuple of (training_successful, comprehensive_results)
        """
        training_start_time = datetime.now()
        logging.info("=" * 60)
        logging.info("STARTING CLUSTERING TRAINING PIPELINE")
        logging.info("=" * 60)
        
        # Reset failure tracking
        self.training_failures.clear()
        self.training_warnings.clear()
        
        # Initialize results
        results = {
            "training_successful": False,
            "validation_results": {},
            "preprocessing_results": {},
            "training_results": {},
            "metrics_results": {},
            "quality_results": {},
            "save_results": {},
            "failures": [],
            "warnings": []
        }
        
        # Stage 1: Validation
        logging.info("Stage 1: Validating training data...")
        is_valid, validation_errors = self.validate_training_data(embedding_matrix, valid_indices)
        results["validation_results"] = {"is_valid": is_valid, "errors": validation_errors}
        
        if not is_valid:
            results["failures"].extend(validation_errors)
            logging.error("✗ Data validation failed: %s", validation_errors)
            return False, results
        
        logging.info("✓ Data validation passed")
          # Validate parameters
        param_valid, param_errors = self.validate_hdbscan_parameters(embedding_matrix, **hdbscan_params)
        if not param_valid:
            results["failures"].extend(param_errors)
            logging.error("✗ Parameter validation failed: %s", param_errors)
            return False, results
        
        # Stage 2: Preprocessing
        logging.info("Stage 2: Preprocessing embeddings...")
        preprocessing_config = preprocessing_config or {"apply_scaling": True, "apply_pca": False}
        
        processed_embeddings, preprocessing_info = self.preprocess_embeddings_for_training(
            embedding_matrix, **preprocessing_config
        )
        results["preprocessing_results"] = preprocessing_info
        
        # Stage 3: Training
        logging.info("Stage 3: Training HDBSCAN model...")
        fallback_params_list = fallback_params_list or []
        
        trained_model, training_info = self.train_hdbscan_with_fallbacks(
            processed_embeddings, hdbscan_params, fallback_params_list
        )
        results["training_results"] = training_info
        
        if trained_model is None:
            results["failures"].append("All training attempts failed")
            logging.error("✗ All training attempts failed")
            return False, results
        
        # Get cluster labels
        cluster_labels = trained_model.labels_
        self.trained_model = trained_model
        self.cluster_labels = cluster_labels
        
        # Stage 4: Calculate metrics
        logging.info("Stage 4: Calculating clustering metrics...")
        metrics, metric_errors = self.calculate_clustering_metrics(processed_embeddings, cluster_labels)
        results["metrics_results"] = {"metrics": metrics, "errors": metric_errors}
        results["warnings"].extend(metric_errors)
          # Stage 5: Quality validation
        logging.info("Stage 5: Validating clustering quality...")
        quality_acceptable, quality_issues = self.validate_clustering_quality(metrics, len(processed_embeddings))
        results["quality_results"] = {"acceptable": quality_acceptable, "issues": quality_issues}
        
        if not quality_acceptable:
            results["warnings"].extend(quality_issues)
            logging.warning("⚠️ Clustering quality below thresholds: %s", quality_issues)
        
        # Stage 6: Save results
        logging.info("Stage 6: Saving training results...")
        save_successful, save_errors = self.save_training_results(
            trained_model, processed_embeddings, cluster_labels, valid_indices, 
            {**metrics, **training_info, **preprocessing_info}, output_dir
        )
        results["save_results"] = {"successful": save_successful, "errors": save_errors}
        
        if save_errors:
            results["warnings"].extend(save_errors)
        
        # Compile final statistics
        training_duration = datetime.now() - training_start_time
        
        self.training_stats = {
            "training_duration_seconds": training_duration.total_seconds(),
            "data_points_trained": len(processed_embeddings),
            "clusters_found": metrics.get("n_clusters", 0),
            "noise_points": metrics.get("n_noise_points", 0),
            "quality_acceptable": quality_acceptable,
            "used_fallback_params": training_info.get("used_fallback", False),
            "preprocessing_applied": preprocessing_info,
            "final_metrics": metrics
        }
        
        results["training_successful"] = True
        results["failures"] = self.training_failures
        results["warnings"] = self.training_warnings
        
        # Log final summary
        self._log_training_summary(results)
        
        return True, results
    
    def _log_training_summary(self, results: Dict):
        """Log comprehensive training summary"""
        logging.info("=" * 60)
        logging.info("TRAINING COMPLETE - SUMMARY")
        logging.info("=" * 60)
        
        if results["training_successful"]:
            metrics = results["metrics_results"]["metrics"]
            logging.info("✅ Training successful!")
            logging.info("  - Clusters found: %d", metrics.get("n_clusters", 0))
            logging.info("  - Noise points: %d (%.1f%%)", 
                        metrics.get("n_noise_points", 0),
                        metrics.get("noise_ratio", 0) * 100)
            
            if "silhouette_score" in metrics:
                logging.info("  - Silhouette score: %.3f", metrics["silhouette_score"])
            
            if results["quality_results"]["acceptable"]:
                logging.info("  - Quality: ✅ Acceptable")
            else:
                logging.warning("  - Quality: ⚠️ Below thresholds")
                
        else:
            logging.error("❌ Training failed")
            
        if results["failures"]:
            logging.error("Failures: %s", results["failures"])
            
        if results["warnings"]:
            logging.warning("Warnings: %s", results["warnings"])
    
    @staticmethod
    def suggest_parameters_for_dataset_size(dataset_size: int) -> Dict:
        """
        Suggest appropriate HDBSCAN parameters based on dataset size.
        
        Args:
            dataset_size: Number of samples in the dataset
            
        Returns:
            Dictionary with suggested primary and fallback parameters
        """
        if dataset_size < 30:
            # Very small datasets
            primary_params = {"min_cluster_size": 3, "min_samples": 1, "metric": "euclidean"}
            fallback_params = [
                {"min_cluster_size": 2, "min_samples": 1, "metric": "cosine"},
                {"min_cluster_size": 2, "min_samples": 1, "metric": "euclidean"}
            ]
        elif dataset_size < 50:
            # Small datasets
            primary_params = {"min_cluster_size": 5, "min_samples": 2, "metric": "euclidean"}
            fallback_params = [
                {"min_cluster_size": 4, "min_samples": 1, "metric": "euclidean"},
                {"min_cluster_size": 3, "min_samples": 1, "metric": "cosine"},
                {"min_cluster_size": 2, "min_samples": 1, "metric": "euclidean"}
            ]
        elif dataset_size < 100:
            # Medium-small datasets
            primary_params = {"min_cluster_size": 8, "min_samples": 3, "metric": "euclidean"}
            fallback_params = [
                {"min_cluster_size": 6, "min_samples": 2, "metric": "euclidean"},
                {"min_cluster_size": 4, "min_samples": 1, "metric": "cosine"},
                {"min_cluster_size": 3, "min_samples": 1, "metric": "euclidean"}
            ]
        elif dataset_size < 200:
            # Medium datasets
            primary_params = {"min_cluster_size": 12, "min_samples": 4, "metric": "euclidean"}
            fallback_params = [
                {"min_cluster_size": 8, "min_samples": 3, "metric": "euclidean"},
                {"min_cluster_size": 6, "min_samples": 2, "metric": "cosine"},
                {"min_cluster_size": 4, "min_samples": 1, "metric": "euclidean"}
            ]
        else:
            # Large datasets
            primary_params = {"min_cluster_size": 15, "min_samples": 5, "metric": "euclidean"}
            fallback_params = [
                {"min_cluster_size": 12, "min_samples": 4, "metric": "euclidean"},
                {"min_cluster_size": 8, "min_samples": 3, "metric": "cosine"},
                {"min_cluster_size": 6, "min_samples": 2, "metric": "euclidean"}
            ]
        
        return {
            "primary_params": primary_params,
            "fallback_params": fallback_params,
            "dataset_size_category": ClusteringTrainer._get_dataset_size_category(dataset_size)
        }
    
    @staticmethod
    def _get_dataset_size_category(dataset_size: int) -> str:
        """Get descriptive category for dataset size"""
        if dataset_size < 30:
            return "very_small"
        elif dataset_size < 50:
            return "small"
        elif dataset_size < 100:
            return "medium_small"
        elif dataset_size < 200:
            return "medium"
        else:
            return "large"