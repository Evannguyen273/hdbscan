# preprocessing/embedding_preprocessor.py
# Updated for new config structure and cumulative training approach
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

from config.config import get_config
from .embedding_generation import EmbeddingGenerator

class EmbeddingProcessor:
    """
    Embedding processor for preprocessing embeddings before clustering.
    Handles normalization, dimensionality reduction, and quality filtering.
    """
    
    def __init__(self, config=None):
        """Initialize embedding processor with updated config system"""
        self.config = config if config is not None else get_config()
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(self.config)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.pca = None
        self.umap_reducer = None
        
        # Processing statistics
        self.processing_stats = {
            "embeddings_processed": 0,
            "embeddings_filtered": 0,
            "dimensionality_reductions": 0,
            "processing_time": 0
        }
        
        # Get preprocessing configuration
        self.preprocessing_config = self._get_preprocessing_config()
        
        logging.info("Embedding processor initialized with config")
    
    def _get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration with defaults"""
        clustering_config = self.config.clustering
        
        return {
            "normalize_embeddings": clustering_config.config.get('normalize_embeddings', True),
            "filter_outliers": clustering_config.config.get('filter_outliers', True),
            "outlier_threshold": clustering_config.config.get('outlier_threshold', 3.0),
            "apply_pca": clustering_config.config.get('apply_pca', False),
            "pca_components": clustering_config.config.get('pca_components', 50),
            "apply_umap": clustering_config.umap.get('enabled', False),
            "umap_n_components": clustering_config.umap.get('n_components', 15),
            "umap_n_neighbors": clustering_config.umap.get('n_neighbors', 15),
            "umap_min_dist": clustering_config.umap.get('min_dist', 0.1),
            "umap_metric": clustering_config.umap.get('metric', 'cosine')
        }
    
    async def process_embeddings_for_training(self, texts: List[str], 
                                            batch_size: int = 50) -> Tuple[np.ndarray, List[int], Dict]:
        """
        Process texts to generate and preprocess embeddings for training.
        
        Args:
            texts: List of texts to process
            batch_size: Batch size for embedding generation
            
        Returns:
            Tuple of (processed_embeddings, valid_indices, processing_stats)
        """
        processing_start = datetime.now()
        
        logging.info("Processing embeddings for training: %d texts", len(texts))
        
        try:
            # Stage 1: Generate embeddings
            embeddings, valid_indices = await self.embedding_generator.generate_embeddings_batch(
                texts, batch_size=batch_size
            )
            
            if len(embeddings) == 0:
                logging.warning("No embeddings generated from texts")
                return np.array([]), [], {"status": "failed", "reason": "no_embeddings"}
            
            # Stage 2: Preprocess embeddings
            processed_embeddings = self._preprocess_embeddings(embeddings)
            
            # Stage 3: Filter outliers if enabled
            if self.preprocessing_config["filter_outliers"]:
                processed_embeddings, outlier_mask = self._filter_outliers(processed_embeddings)
                # Update valid indices to account for filtered outliers
                valid_indices = [valid_indices[i] for i, keep in enumerate(outlier_mask) if keep]
            
            # Stage 4: Apply dimensionality reduction if configured
            if self.preprocessing_config["apply_pca"] and len(processed_embeddings) > 0:
                processed_embeddings = self._apply_pca(processed_embeddings)
            
            if self.preprocessing_config["apply_umap"] and len(processed_embeddings) > 0:
                processed_embeddings = self._apply_umap(processed_embeddings)
            
            # Update statistics
            processing_duration = datetime.now() - processing_start
            self.processing_stats["embeddings_processed"] += len(processed_embeddings)
            self.processing_stats["processing_time"] += processing_duration.total_seconds()
            
            stats = {
                "status": "success",
                "original_texts": len(texts),
                "embeddings_generated": len(embeddings),
                "embeddings_after_preprocessing": len(processed_embeddings),
                "valid_indices_count": len(valid_indices),
                "processing_duration_seconds": processing_duration.total_seconds(),
                "preprocessing_config_used": self.preprocessing_config
            }
            
            logging.info("Embedding processing completed: %d final embeddings from %d texts",
                        len(processed_embeddings), len(texts))
            
            return processed_embeddings, valid_indices, stats
            
        except Exception as e:
            processing_duration = datetime.now() - processing_start
            logging.error("Embedding processing failed: %s", str(e))
            
            stats = {
                "status": "failed",
                "error": str(e),
                "processing_duration_seconds": processing_duration.total_seconds()
            }
            
            return np.array([]), [], stats
    
    async def process_embeddings_for_prediction(self, texts: List[str]) -> Tuple[np.ndarray, List[int], Dict]:
        """
        Process texts for real-time prediction (optimized for speed).
        
        Args:
            texts: List of texts to process
            
        Returns:
            Tuple of (processed_embeddings, valid_indices, processing_stats)
        """
        processing_start = datetime.now()
        
        try:
            # Use smaller batch size for faster prediction
            batch_size = min(25, len(texts))
            
            # Generate embeddings
            embeddings, valid_indices = await self.embedding_generator.generate_embeddings_batch(
                texts, batch_size=batch_size
            )
            
            if len(embeddings) == 0:
                return np.array([]), [], {"status": "failed", "reason": "no_embeddings"}
            
            # Apply only essential preprocessing for prediction speed
            processed_embeddings = self._preprocess_embeddings(embeddings, training_mode=False)
            
            processing_duration = datetime.now() - processing_start
            
            stats = {
                "status": "success",
                "embeddings_generated": len(embeddings),
                "embeddings_processed": len(processed_embeddings),
                "processing_duration_seconds": processing_duration.total_seconds()
            }
            
            return processed_embeddings, valid_indices, stats
            
        except Exception as e:
            processing_duration = datetime.now() - processing_start
            logging.error("Prediction embedding processing failed: %s", str(e))
            
            return np.array([]), [], {
                "status": "failed", 
                "error": str(e),
                "processing_duration_seconds": processing_duration.total_seconds()
            }
    
    def _preprocess_embeddings(self, embeddings: np.ndarray, training_mode: bool = True) -> np.ndarray:
        """Apply basic preprocessing to embeddings"""
        processed = embeddings.copy()
        
        # Normalize embeddings if configured
        if self.preprocessing_config["normalize_embeddings"]:
            # L2 normalization
            norms = np.linalg.norm(processed, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            processed = processed / norms
            
            logging.debug("Applied L2 normalization to embeddings")
        
        # Standardize embeddings (optional)
        if training_mode and self.preprocessing_config.get("standardize_embeddings", False):
            processed = self.scaler.fit_transform(processed)
            logging.debug("Applied standardization to embeddings")
        elif not training_mode and hasattr(self.scaler, 'scale_'):
            # Use fitted scaler for prediction
            processed = self.scaler.transform(processed)
        
        return processed
    
    def _filter_outliers(self, embeddings: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
        """Filter out outlier embeddings based on distance from centroid"""
        if len(embeddings) == 0:
            return embeddings, []
        
        try:
            # Calculate distances from centroid
            centroid = np.mean(embeddings, axis=0)
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            
            # Identify outliers using threshold
            threshold = self.preprocessing_config["outlier_threshold"]
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            outlier_threshold = mean_distance + (threshold * std_distance)
            
            # Create mask for non-outliers
            outlier_mask = distances <= outlier_threshold
            filtered_embeddings = embeddings[outlier_mask]
            
            outliers_removed = len(embeddings) - len(filtered_embeddings)
            if outliers_removed > 0:
                logging.info("Filtered out %d outlier embeddings (%.1f%%)", 
                           outliers_removed, outliers_removed / len(embeddings) * 100)
                self.processing_stats["embeddings_filtered"] += outliers_removed
            
            return filtered_embeddings, outlier_mask.tolist()
            
        except Exception as e:
            logging.error("Outlier filtering failed: %s", str(e))
            return embeddings, [True] * len(embeddings)
    
    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA dimensionality reduction"""
        if len(embeddings) == 0:
            return embeddings
        
        try:
            n_components = min(
                self.preprocessing_config["pca_components"],
                embeddings.shape[0] - 1,
                embeddings.shape[1]
            )
            
            if n_components < embeddings.shape[1]:
                if self.pca is None:
                    self.pca = PCA(n_components=n_components, random_state=42)
                    reduced_embeddings = self.pca.fit_transform(embeddings)
                    
                    explained_variance = np.sum(self.pca.explained_variance_ratio_)
                    logging.info("Applied PCA: %d -> %d dimensions (%.1f%% variance retained)",
                               embeddings.shape[1], n_components, explained_variance * 100)
                else:
                    reduced_embeddings = self.pca.transform(embeddings)
                
                self.processing_stats["dimensionality_reductions"] += 1
                return reduced_embeddings
            else:
                logging.debug("PCA skipped: requested components >= current dimensions")
                return embeddings
                
        except Exception as e:
            logging.error("PCA reduction failed: %s", str(e))
            return embeddings
    
    def _apply_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP dimensionality reduction"""
        if len(embeddings) == 0:
            return embeddings
        
        try:
            n_components = min(
                self.preprocessing_config["umap_n_components"],
                embeddings.shape[1]
            )
            
            if n_components < embeddings.shape[1] and len(embeddings) > self.preprocessing_config["umap_n_neighbors"]:
                if self.umap_reducer is None:
                    self.umap_reducer = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=self.preprocessing_config["umap_n_neighbors"],
                        min_dist=self.preprocessing_config["umap_min_dist"],
                        metric=self.preprocessing_config["umap_metric"],
                        random_state=42
                    )
                    reduced_embeddings = self.umap_reducer.fit_transform(embeddings)
                    
                    logging.info("Applied UMAP: %d -> %d dimensions",
                               embeddings.shape[1], n_components)
                else:
                    reduced_embeddings = self.umap_reducer.transform(embeddings)
                
                self.processing_stats["dimensionality_reductions"] += 1
                return reduced_embeddings
            else:
                logging.debug("UMAP skipped: insufficient data or dimensions")
                return embeddings
                
        except Exception as e:
            logging.error("UMAP reduction failed: %s", str(e))
            return embeddings
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get embedding processing statistics"""
        stats = self.processing_stats.copy()
        
        # Add embedding generation stats
        gen_stats = self.embedding_generator.get_generation_statistics()
        stats.update({f"generation_{k}": v for k, v in gen_stats.items()})
        
        return stats
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate embedding processor configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Validate embedding generation
        gen_validation = self.embedding_generator.validate_configuration()
        validation_results["warnings"].extend(gen_validation["warnings"])
        validation_results["errors"].extend(gen_validation["errors"])
        
        if not gen_validation["valid"]:
            validation_results["valid"] = False
        
        # Validate preprocessing configuration
        if self.preprocessing_config["apply_pca"] and self.preprocessing_config["pca_components"] <= 0:
            validation_results["errors"].append("PCA components must be positive")
            validation_results["valid"] = False
        
        if self.preprocessing_config["apply_umap"]:
            if self.preprocessing_config["umap_n_components"] <= 0:
                validation_results["errors"].append("UMAP components must be positive")
                validation_results["valid"] = False
            
            if self.preprocessing_config["umap_n_neighbors"] <= 0:
                validation_results["errors"].append("UMAP n_neighbors must be positive")
                validation_results["valid"] = False
        
        return validation_results
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.processing_stats = {
            "embeddings_processed": 0,
            "embeddings_filtered": 0,
            "dimensionality_reductions": 0,
            "processing_time": 0
        }
        self.embedding_generator.reset_statistics()
        logging.info("Embedding processor statistics reset")