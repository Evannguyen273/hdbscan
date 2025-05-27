import logging
import numpy as np
import pandas as pd
import json
import pickle
import os
import time
from datetime import datetime
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import uuid

class HDBSCANClusterer:
    def __init__(self, config):
        self.config = config
    
    def train_hdbscan(
        self, 
        df_with_embeddings: pd.DataFrame,
        dataset_name: str,
        result_path: str,
        use_checkpoint: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray, object, object]:
        """Train HDBSCAN clustering model on embeddings with UMAP checkpoint"""
        
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        # Create output directory
        output_dir = f"{result_path}/{dataset_name}/clustering"
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Training HDBSCAN for dataset '{dataset_name}'")
        
        # Define checkpoint paths
        umap_checkpoint_path = f"{output_dir}/umap_embeddings_checkpoint.npy"
        umap_reducer_checkpoint_path = f"{output_dir}/umap_reducer_checkpoint.pkl"
        
        # Check if UMAP checkpoint exists
        umap_embeddings = None
        reducer = None
        use_existing_checkpoint = (use_checkpoint and 
                                  os.path.exists(umap_checkpoint_path) and 
                                  os.path.exists(umap_reducer_checkpoint_path))
        
        if use_existing_checkpoint:
            try:
                logging.info(f"Loading UMAP checkpoint from {umap_checkpoint_path}")
                umap_embeddings = np.load(umap_checkpoint_path)
                
                logging.info(f"Loading UMAP reducer from {umap_reducer_checkpoint_path}")
                with open(umap_reducer_checkpoint_path, 'rb') as f:
                    reducer = pickle.load(f)
                    
                logging.info(f"Successfully loaded UMAP checkpoint with {umap_embeddings.shape[1]} dimensions")
            except Exception as e:
                logging.error(f"Error loading UMAP checkpoint: {e}. Will recompute UMAP projection.")
                umap_embeddings = None
                reducer = None
        
        # If no checkpoint was loaded, run the UMAP projection
        if umap_embeddings is None:
            # Extract embeddings from the dataframe
            logging.info("Extracting embeddings...")
            embeddings = np.array([json.loads(emb) for emb in df_with_embeddings['embedding'].tolist()])
            
            # Standardize embeddings
            logging.info("Standardizing embeddings...")
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)
            
            # Apply UMAP for dimensionality reduction
            logging.info(f"Applying UMAP to reduce dimensions from {scaled_embeddings.shape[1]} to {self.config.clustering.umap_n_components}...")
            reducer = umap.UMAP(
                n_components=self.config.clustering.umap_n_components,
                n_neighbors=self.config.clustering.umap_n_neighbors,
                min_dist=self.config.clustering.umap_min_dist,
                metric='cosine',
                random_state=42,
                low_memory=False,
                spread=1.0,
                local_connectivity=2,
                verbose=True
            )
            
            umap_embeddings = reducer.fit_transform(scaled_embeddings)
            
            # Save checkpoint if checkpointing is enabled
            if use_checkpoint:
                try:
                    logging.info(f"Saving UMAP embeddings checkpoint to {umap_checkpoint_path}")
                    np.save(umap_checkpoint_path, umap_embeddings)
                    
                    logging.info(f"Saving UMAP reducer to {umap_reducer_checkpoint_path}")
                    with open(umap_reducer_checkpoint_path, 'wb') as f:
                        pickle.dump(reducer, f)
                        
                    logging.info("UMAP checkpoint saved successfully")
                except Exception as e:
                    logging.error(f"Error saving UMAP checkpoint: {e}")
        
        # Apply HDBSCAN for clustering
        logging.info(f"Applying HDBSCAN with min_cluster_size={self.config.clustering.min_cluster_size}, min_samples={self.config.clustering.min_samples}...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.clustering.min_cluster_size,
            min_samples=self.config.clustering.min_samples,
            metric='euclidean',
            cluster_selection_method='leaf',
            cluster_selection_epsilon=0.2,
            prediction_data=True,
            core_dist_n_jobs=-1
        )
        
        cluster_labels = clusterer.fit_predict(umap_embeddings)
        
        # Add cluster labels to dataframe
        result_df = df_with_embeddings.copy()
        result_df['cluster'] = cluster_labels
        
        # Add cluster probabilities
        if hasattr(clusterer, 'probabilities_'):
            result_df['cluster_probability'] = clusterer.probabilities_
        
        # Count clusters and noise points
        clusters = set(cluster_labels)
        n_clusters = len(clusters) - (1 if -1 in clusters else 0)
        n_noise = list(cluster_labels).count(-1)
        noise_percentage = 100 * n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
        
        logging.info(f"HDBSCAN found {n_clusters} clusters")
        logging.info(f"Noise points: {n_noise} ({noise_percentage:.2f}% of data)")
        
        # Save results
        result_df.to_parquet(f"{output_dir}/clustered_df.parquet", index=False)
        
        # Save UMAP embeddings for subsequent analysis
        try:
            np.save(f"{output_dir}/umap_embeddings.npy", umap_embeddings)
        except Exception as e:
            logging.warning(f"Could not save final UMAP embeddings: {e}")
        
        # Save models
        with open(f"{output_dir}/umap_reducer.pkl", "wb") as f:
            pickle.dump(reducer, f)
        
        with open(f"{output_dir}/hdbscan_clusterer.pkl", "wb") as f:
            pickle.dump(clusterer, f)
        
        # Create metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "min_cluster_size": self.config.clustering.min_cluster_size,
                "min_samples": self.config.clustering.min_samples,
                "umap_n_components": self.config.clustering.umap_n_components,
                "umap_n_neighbors": self.config.clustering.umap_n_neighbors,
                "umap_min_dist": self.config.clustering.umap_min_dist
            },
            "results": {
                "num_clusters": n_clusters,
                "noise_points": n_noise,
                "noise_percentage": noise_percentage
            },
            "runtime_seconds": time.time() - start_time,
            "dataset_name": dataset_name,
            "correlation_id": correlation_id,
            "checkpoint_used": use_existing_checkpoint
        }
        
        # Save metadata
        with open(f"{output_dir}/clustering_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        total_time = time.time() - start_time
        logging.info(f"HDBSCAN training completed in {total_time:.2f} seconds")
        
        return result_df, umap_embeddings, clusterer, reducer