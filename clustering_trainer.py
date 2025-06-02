#!/usr/bin/env python3
"""
Clustering Trainer - Handles HDBSCAN model training with versioned outputs
Integrates with the cumulative training approach
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# ML imports
import hdbscan
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Local imports
from config.config import get_config
from logging_setup import setup_detailed_logging


class ClusteringTrainer:
    """
    Handles HDBSCAN clustering training with versioned table outputs
    Supports cumulative 24-month training approach
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the clustering trainer"""
        setup_detailed_logging(logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.config = get_config(config_path)
        
        # Create output directories
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        
        self.logger.info("Clustering Trainer initialized for cumulative training")
    
    def train_hdbscan_model(self, embeddings: np.ndarray, tech_center: str, 
                           year: int, quarter: str) -> Dict[str, Any]:
        """
        Train HDBSCAN model with cumulative 24-month data
        
        Args:
            embeddings: Document embeddings (24-month cumulative dataset)
            tech_center: Name of the tech center
            year: Training year
            quarter: Training quarter
            
        Returns:
            Dictionary with training results and versioned model info
        """
        try:
            self.logger.info(f"Training HDBSCAN model for {tech_center} - {year} Q{quarter}")
            self.logger.info(f"Input data: {len(embeddings)} incidents (24-month cumulative)")
            
            # 1. Apply UMAP dimensionality reduction
            umap_results = self._apply_umap_reduction(embeddings, tech_center)
            
            # 2. Train HDBSCAN clustering
            clustering_results = self._train_hdbscan_clustering(
                umap_results["umap_embeddings"], tech_center
            )
            
            # 3. Evaluate clustering performance
            evaluation_metrics = self._evaluate_clustering(
                umap_results["umap_embeddings"], 
                clustering_results["cluster_labels"]
            )
            
            # 4. Save versioned models
            model_paths = self._save_versioned_models(
                {**umap_results, **clustering_results}, 
                tech_center, year, quarter
            )
            
            # 5. Prepare training summary with versioning info
            training_summary = {
                "tech_center": tech_center,
                "training_year": year,
                "training_quarter": quarter,
                "training_approach": "cumulative_24_months",
                "data_points": len(embeddings),
                "clusters_found": len(np.unique(clustering_results["cluster_labels"])),
                "noise_points": np.sum(clustering_results["cluster_labels"] == -1),
                "evaluation_metrics": evaluation_metrics,
                "model_paths": model_paths,
                "model_version": f"{year}_{quarter}_v1",
                "training_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"HDBSCAN training completed: {training_summary['clusters_found']} clusters")
            return {**umap_results, **clustering_results, "training_summary": training_summary}
            
        except Exception as e:
            self.logger.error(f"HDBSCAN training failed for {tech_center}: {e}")
            raise
    
    def _apply_umap_reduction(self, embeddings: np.ndarray, tech_center: str) -> Dict[str, Any]:
        """Apply UMAP dimensionality reduction"""
        self.logger.info(f"Applying UMAP reduction for {tech_center}")
        
        # UMAP parameters optimized for incident clustering
        umap_model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        
        # Fit and transform embeddings
        umap_embeddings = umap_model.fit_transform(embeddings)
        
        self.logger.info(f"UMAP reduction completed: {embeddings.shape} -> {umap_embeddings.shape}")
        
        return {
            "umap_model": umap_model,
            "umap_embeddings": umap_embeddings,
            "original_embeddings": embeddings
        }
    
    def _train_hdbscan_clustering(self, umap_embeddings: np.ndarray, tech_center: str) -> Dict[str, Any]:
        """Train HDBSCAN clustering model"""
        self.logger.info(f"Training HDBSCAN clustering for {tech_center}")
        
        # HDBSCAN parameters optimized for incident data
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=max(5, len(umap_embeddings) // 200),  # Adaptive min cluster size
            min_samples=3,
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom',
            algorithm='best',
            memory='auto'
        )
        
        # Fit clustering model
        cluster_labels = hdbscan_model.fit_predict(umap_embeddings)
        
        # Calculate additional metrics
        cluster_probabilities = hdbscan_model.probabilities_
        outlier_scores = hdbscan_model.outlier_scores_
        
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        self.logger.info(f"HDBSCAN clustering completed: {n_clusters} clusters, {np.sum(cluster_labels == -1)} noise points")
        
        return {
            "hdbscan_model": hdbscan_model,
            "cluster_labels": cluster_labels,
            "cluster_probabilities": cluster_probabilities,
            "outlier_scores": outlier_scores,
            "n_clusters": n_clusters
        }
    
    def _evaluate_clustering(self, umap_embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering performance"""
        try:
            # Filter out noise points for evaluation
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) < 2:
                return {"silhouette_score": 0.0, "calinski_harabasz_score": 0.0}
            
            filtered_embeddings = umap_embeddings[non_noise_mask]
            filtered_labels = cluster_labels[non_noise_mask]
            
            # Calculate silhouette score
            if len(np.unique(filtered_labels)) > 1:
                sil_score = silhouette_score(filtered_embeddings, filtered_labels)
                ch_score = calinski_harabasz_score(filtered_embeddings, filtered_labels)
            else:
                sil_score = 0.0
                ch_score = 0.0
            
            metrics = {
                "silhouette_score": float(sil_score),
                "calinski_harabasz_score": float(ch_score),
                "noise_ratio": float(np.sum(cluster_labels == -1) / len(cluster_labels))
            }
            
            self.logger.info(f"Clustering evaluation: Silhouette={sil_score:.3f}, CH={ch_score:.1f}")
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Clustering evaluation failed: {e}")
            return {"silhouette_score": 0.0, "calinski_harabasz_score": 0.0, "noise_ratio": 1.0}
      def _save_versioned_models(self, model_results: Dict[str, Any], 
                             tech_center: str, year: int, quarter: str) -> Dict[str, str]:
        """
        Save models with version information to both local storage and blob storage
        
        Storage Strategy:
        1. Local storage: For development and immediate access
        2. Blob storage: For production deployment and prediction pipeline
        """
        tech_center_safe = tech_center.replace(" ", "_").replace("-", "_")
        model_version = f"{year}_{quarter}"
        
        # Create local model directory
        model_dir = self.models_path / tech_center_safe / model_version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_paths = {}
        
        try:
            # Save models locally first
            local_paths = self._save_models_locally(model_results, model_dir, tech_center, model_version)
            model_paths.update(local_paths)
            
            # Upload to blob storage for production use
            blob_paths = self._upload_models_to_blob_storage(
                local_paths, tech_center, year, quarter, model_version
            )
            model_paths.update(blob_paths)
            
            self.logger.info(f"Models saved locally to {model_dir}")
            self.logger.info(f"Models uploaded to blob storage: {blob_paths.get('blob_container', 'N/A')}")
            
            return model_paths
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            raise
    
    def _save_models_locally(self, model_results: Dict[str, Any], model_dir: Path, 
                           tech_center: str, model_version: str) -> Dict[str, str]:
        """Save models to local file system"""
        local_paths = {}
        
        # Save UMAP model
        umap_path = model_dir / "umap_model.pkl"
        with open(umap_path, 'wb') as f:
            pickle.dump(model_results["umap_model"], f)
        local_paths["umap_model_local"] = str(umap_path)
        
        # Save HDBSCAN model
        hdbscan_path = model_dir / "hdbscan_model.pkl"
        with open(hdbscan_path, 'wb') as f:
            pickle.dump(model_results["hdbscan_model"], f)
        local_paths["hdbscan_model_local"] = str(hdbscan_path)
        
        # Save UMAP embeddings
        embeddings_path = model_dir / "umap_embeddings.npy"
        np.save(embeddings_path, model_results["umap_embeddings"])
        local_paths["umap_embeddings_local"] = str(embeddings_path)
        
        # Save cluster labels
        labels_path = model_dir / "cluster_labels.npy"
        np.save(labels_path, model_results["cluster_labels"])
        local_paths["cluster_labels_local"] = str(labels_path)
        
        # Save model metadata
        metadata = {
            "tech_center": tech_center,
            "model_version": model_version,
            "training_timestamp": datetime.now().isoformat(),
            "n_clusters": model_results["n_clusters"],
            "data_points": len(model_results["cluster_labels"]),
            "training_approach": "cumulative_24_months",
            "storage_locations": {
                "local": str(model_dir),
                "blob_container": f"hdbscan-models/{tech_center.replace(' ', '-').lower()}",
                "blob_prefix": f"{model_version}/"
            }
        }
        
        metadata_path = model_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        local_paths["metadata_local"] = str(metadata_path)
        
        return local_paths
    
    def _upload_models_to_blob_storage(self, local_paths: Dict[str, str], 
                                     tech_center: str, year: int, quarter: str, 
                                     model_version: str) -> Dict[str, str]:
        """
        Upload trained models to Azure Blob Storage for production use
        
        Blob Storage Structure:
        hdbscan-models/
        ├── bt-tc-data-analytics/
        │   ├── 2024_q4/
        │   │   ├── umap_model.pkl
        │   │   ├── hdbscan_model.pkl
        │   │   ├── umap_embeddings.npy
        │   │   ├── cluster_labels.npy
        │   │   └── model_metadata.json
        │   └── 2025_q2/
        │       └── ...
        """
        try:
            # Mock blob storage upload (in real implementation, use Azure Blob Storage SDK)
            container_name = "hdbscan-models"
            tech_center_folder = tech_center.replace(" ", "-").replace("_", "-").lower()
            blob_prefix = f"{tech_center_folder}/{model_version}/"
            
            blob_paths = {
                "blob_container": container_name,
                "blob_prefix": blob_prefix,
                "umap_model_blob": f"{blob_prefix}umap_model.pkl",
                "hdbscan_model_blob": f"{blob_prefix}hdbscan_model.pkl",
                "umap_embeddings_blob": f"{blob_prefix}umap_embeddings.npy",
                "cluster_labels_blob": f"{blob_prefix}cluster_labels.npy",
                "metadata_blob": f"{blob_prefix}model_metadata.json"
            }
            
            # In real implementation, upload each file:
            # blob_service_client = BlobServiceClient(...)
            # for local_path, blob_path in upload_mapping.items():
            #     with open(local_path, 'rb') as data:
            #         blob_service_client.get_blob_client(
            #             container=container_name, blob=blob_path
            #         ).upload_blob(data, overwrite=True)
            
            self.logger.info(f"Models would be uploaded to: {container_name}/{blob_prefix}")
            return blob_paths
            
        except Exception as e:
            self.logger.warning(f"Failed to upload to blob storage: {e}")
            return {"blob_upload_status": "failed", "error": str(e)}


class ModelVersionManager:
    """Manages versioned model storage and retrieval"""
    
    def __init__(self, models_path: Path):
        self.models_path = models_path
        self.logger = logging.getLogger(__name__)
    
    def get_latest_model_version(self, tech_center: str) -> Optional[str]:
        """Get the latest model version for a tech center"""
        tech_center_safe = tech_center.replace(" ", "_").replace("-", "_")
        tech_center_path = self.models_path / tech_center_safe
        
        if not tech_center_path.exists():
            return None
        
        # Get all version directories
        version_dirs = [d for d in tech_center_path.iterdir() if d.is_dir()]
        if not version_dirs:
            return None
        
        # Sort by modification time (latest first)
        version_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return version_dirs[0].name
      def load_model(self, tech_center: str, version: str) -> Dict[str, Any]:
        """
        Load a specific model version from blob storage (production) or local storage (development)
        
        Priority:
        1. Try blob storage first (production deployment)
        2. Fallback to local storage (development/testing)
        """
        try:
            # First, try to load from blob storage
            model_data = self._load_model_from_blob_storage(tech_center, version)
            if model_data:
                self.logger.info(f"Loaded model {version} for {tech_center} from blob storage")
                return model_data
        except Exception as e:
            self.logger.warning(f"Failed to load from blob storage: {e}")
        
        # Fallback to local storage
        try:
            model_data = self._load_model_from_local_storage(tech_center, version)
            self.logger.info(f"Loaded model {version} for {tech_center} from local storage")
            return model_data
        except Exception as e:
            raise RuntimeError(f"Failed to load model from both blob and local storage: {e}")
    
    def _load_model_from_blob_storage(self, tech_center: str, version: str) -> Optional[Dict[str, Any]]:
        """Load model from Azure Blob Storage"""
        try:
            # Mock blob storage download (in real implementation, use Azure Blob Storage SDK)
            container_name = "hdbscan-models"
            tech_center_folder = tech_center.replace(" ", "-").replace("_", "-").lower()
            blob_prefix = f"{tech_center_folder}/{version}/"
            
            # In real implementation:
            # blob_service_client = BlobServiceClient(...)
            # umap_blob = blob_service_client.get_blob_client(
            #     container=container_name, blob=f"{blob_prefix}umap_model.pkl"
            # )
            # umap_data = umap_blob.download_blob().readall()
            # umap_model = pickle.loads(umap_data)
            
            self.logger.info(f"Would download model from blob storage: {container_name}/{blob_prefix}")
            return None  # Return None to trigger local fallback in mock
            
        except Exception as e:
            self.logger.error(f"Blob storage download failed: {e}")
            return None
    
    def _load_model_from_local_storage(self, tech_center: str, version: str) -> Dict[str, Any]:
        """Load model from local file system"""
        tech_center_safe = tech_center.replace(" ", "_").replace("-", "_")
        model_dir = self.models_path / tech_center_safe / version
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        
        # Load models
        with open(model_dir / "umap_model.pkl", 'rb') as f:
            umap_model = pickle.load(f)
        
        with open(model_dir / "hdbscan_model.pkl", 'rb') as f:
            hdbscan_model = pickle.load(f)
        
        # Load embeddings and labels
        umap_embeddings = np.load(model_dir / "umap_embeddings.npy")
        cluster_labels = np.load(model_dir / "cluster_labels.npy")
        
        # Load metadata
        with open(model_dir / "model_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return {
            "umap_model": umap_model,
            "hdbscan_model": hdbscan_model,
            "umap_embeddings": umap_embeddings,
            "cluster_labels": cluster_labels,
            "metadata": metadata
        }


def main():
    """Example usage of clustering trainer"""
    trainer = ClusteringTrainer()
    
    # Mock embeddings (in real use, load from preprocessed_incidents table)
    mock_embeddings = np.random.rand(1000, 1536)  # 1000 incidents, 1536-dim embeddings
    
    # Train model
    results = trainer.train_hdbscan_model(
        embeddings=mock_embeddings,
        tech_center="BT-TC-Data Analytics",
        year=2024,
        quarter="q4"
    )
    
    print("Training completed!")
    print(f"Clusters found: {results['training_summary']['clusters_found']}")
    print(f"Model version: {results['training_summary']['model_version']}")


if __name__ == "__main__":
    main()