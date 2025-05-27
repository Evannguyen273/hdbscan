import logging
import time
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.bigquery_client import BigQueryClient
from data.blob_storage import BlobStorageClient
from core.clustering import HDBSCANClusterer
from analysis.cluster_analysis import ClusterAnalyzer
from analysis.cluster_labeling import ClusterLabeler
from analysis.domain_grouping import DomainGrouper

class TechCenterTrainingPipeline:
    """
    Training pipeline for quarterly model retraining per tech center
    Supports parallel training and model artifact management
    """
    
    def __init__(self, config):
        self.config = config
        self.bigquery_client = BigQueryClient(config)
        self.blob_storage = BlobStorageClient(config)
        self.clusterer = HDBSCANClusterer(config)
        self.cluster_analyzer = ClusterAnalyzer(config)
        self.cluster_labeler = ClusterLabeler(config)
        self.domain_grouper = DomainGrouper(config)
        
        # Table names
        self.preprocessed_table = f"{config.bigquery.project_id}.{config.bigquery.datasets.preprocessing}.{config.bigquery.tables.preprocessed}"
        
    def get_current_quarter(self) -> Tuple[int, str]:
        """Get current year and quarter"""
        now = datetime.now()
        year = now.year
        month = now.month
        
        if month in [1, 2, 3]:
            quarter = "q1"
        elif month in [4, 5, 6]:
            quarter = "q2" 
        elif month in [7, 8, 9]:
            quarter = "q3"
        else:
            quarter = "q4"
            
        return year, quarter
    
    def get_training_data(self, tech_center: str, year: int, quarter: str) -> pd.DataFrame:
        """Get training data for a tech center and quarter"""
        
        # Get month range for quarter
        quarter_months = self.config.pipeline.training_schedule.months[quarter]
        start_month = min(quarter_months)
        end_month = max(quarter_months)
        
        query = f"""
        SELECT 
            number,
            sys_created_on,
            combined_incidents_summary,
            embedding,
            tech_center,
            processed_at
        FROM `{self.preprocessed_table}`
        WHERE tech_center = '{tech_center}'
            AND EXTRACT(YEAR FROM sys_created_on) = {year}
            AND EXTRACT(MONTH FROM sys_created_on) BETWEEN {start_month} AND {end_month}
        ORDER BY sys_created_on
        """
        
        logging.info(f"Getting training data for {tech_center} - {year} {quarter}")
        df = self.bigquery_client.run_query(query)
        logging.info(f"Loaded {len(df)} training records for {tech_center}")
        
        return df
    
    def train_tech_center_model(
        self, 
        tech_center: str, 
        year: int, 
        quarter: str,
        save_artifacts: bool = True
    ) -> Dict:
        """Train clustering model for a specific tech center and quarter"""
        
        start_time = time.time()
        tech_center_clean = tech_center.replace(" ", "_").replace("-", "_")
        
        logging.info(f"Starting training for {tech_center} - {year} {quarter}")
        
        try:
            # Get training data
            training_data = self.get_training_data(tech_center, year, quarter)
            
            if training_data.empty:
                return {
                    "tech_center": tech_center,
                    "year": year,
                    "quarter": quarter,
                    "status": "skipped",
                    "reason": "No training data available",
                    "runtime_seconds": time.time() - start_time
                }
            
            if len(training_data) < self.config.clustering.hdbscan.min_cluster_size:
                return {
                    "tech_center": tech_center,
                    "year": year,
                    "quarter": quarter,
                    "status": "skipped",
                    "reason": f"Insufficient data: {len(training_data)} < {self.config.clustering.hdbscan.min_cluster_size}",
                    "runtime_seconds": time.time() - start_time
                }
            
            # Create output directory structure
            if self.config.pipeline.save_to_local:
                local_output_dir = f"{self.config.pipeline.result_path}/models/{tech_center_clean}/{year}/{quarter}"
                os.makedirs(f"{local_output_dir}/embeddings", exist_ok=True)
                os.makedirs(f"{local_output_dir}/clustering", exist_ok=True)
                os.makedirs(f"{local_output_dir}/analysis", exist_ok=True)
                os.makedirs(f"{local_output_dir}/metadata", exist_ok=True)
            
            # Train HDBSCAN clustering
            dataset_name = f"{tech_center_clean}_{year}_{quarter}"
            clustered_df, umap_embeddings, clusterer, reducer = self.clusterer.train_hdbscan(
                df_with_embeddings=training_data,
                dataset_name=dataset_name,
                result_path=self.config.pipeline.result_path,
                use_checkpoint=self.config.pipeline.use_checkpointing
            )
            
            # Generate cluster analysis
            clusters_info = self.cluster_analyzer.generate_cluster_info(
                clustered_df,
                text_column='combined_incidents_summary',
                cluster_column='cluster',
                sample_size=5
            )
            
            # Label clusters with LLM
            try:
                labeled_clusters = self.cluster_labeler.label_clusters_with_llm(
                    clusters_info,
                    max_samples=300,
                    chunk_size=25
                )
            except Exception as e:
                logging.warning(f"LLM labeling failed for {tech_center}, using fallback: {e}")
                labeled_clusters = {
                    str(cid): {"topic": f"Cluster {cid}", "description": "Auto-generated label"}
                    for cid in clustered_df['cluster'].unique() if cid != -1
                }
                labeled_clusters["-1"] = {"topic": "Noise", "description": "Unclustered incidents"}
            
            # Group into domains
            try:
                domains = self.domain_grouper.group_clusters_into_domains(
                    labeled_clusters,
                    clusters_info,
                    umap_embeddings,
                    clustered_df['cluster'].values
                )
            except Exception as e:
                logging.warning(f"Domain grouping failed for {tech_center}, using fallback: {e}")
                domains = {
                    "domains": [
                        {"domain_name": "General", "clusters": list(map(int, clusters_info.keys()))}
                    ]
                }
            
            # Save model artifacts
            if save_artifacts:
                self.save_model_artifacts(
                    tech_center=tech_center,
                    year=year,
                    quarter=quarter,
                    clusterer=clusterer,
                    reducer=reducer,
                    labeled_clusters=labeled_clusters,
                    domains=domains,
                    clusters_info=clusters_info,
                    training_data_size=len(training_data)
                )
            
            runtime = time.time() - start_time
            
            # Calculate metrics
            num_clusters = len([k for k in clusters_info.keys() if k != "-1"])
            noise_percentage = clusters_info.get("-1", {}).get("percentage", 0)
            
            result = {
                "tech_center": tech_center,
                "year": year, 
                "quarter": quarter,
                "status": "success",
                "metrics": {
                    "training_samples": len(training_data),
                    "num_clusters": num_clusters,
                    "num_domains": len(domains.get("domains", [])),
                    "noise_percentage": noise_percentage
                },
                "runtime_seconds": runtime
            }
            
            logging.info(f"Training completed for {tech_center} - {num_clusters} clusters, {runtime:.2f}s")
            return result
            
        except Exception as e:
            logging.error(f"Training failed for {tech_center}: {e}")
            return {
                "tech_center": tech_center,
                "year": year,
                "quarter": quarter,
                "status": "failed",
                "error": str(e),
                "runtime_seconds": time.time() - start_time
            }
    
    def save_model_artifacts(
        self,
        tech_center: str,
        year: int,
        quarter: str,
        clusterer,
        reducer,
        labeled_clusters: Dict,
        domains: Dict,
        clusters_info: Dict,
        training_data_size: int
    ):
        """Save model artifacts to local and blob storage"""
        
        tech_center_clean = tech_center.replace(" ", "_").replace("-", "_")
        
        # Local storage
        if self.config.pipeline.save_to_local:
            base_path = f"{self.config.pipeline.result_path}/models/{tech_center_clean}/{year}/{quarter}"
            
            # Save clustering models
            with open(f"{base_path}/clustering/hdbscan_clusterer.pkl", "wb") as f:
                pickle.dump(clusterer, f)
            
            with open(f"{base_path}/clustering/umap_reducer.pkl", "wb") as f:
                pickle.dump(reducer, f)
            
            # Save analysis results
            with open(f"{base_path}/analysis/labeled_clusters.json", "w") as f:
                json.dump(labeled_clusters, f, indent=2)
            
            with open(f"{base_path}/analysis/domains.json", "w") as f:
                json.dump(domains, f, indent=2)
            
            with open(f"{base_path}/analysis/clusters_info.json", "w") as f:
                json.dump(clusters_info, f, indent=2)
            
            # Save metadata
            metadata = {
                "tech_center": tech_center,
                "year": year,
                "quarter": quarter,
                "training_timestamp": datetime.now().isoformat(),
                "training_data_size": training_data_size,
                "model_version": "1.0",
                "config": {
                    "min_cluster_size": self.config.clustering.hdbscan.min_cluster_size,
                    "min_samples": self.config.clustering.hdbscan.min_samples,
                    "umap_n_components": self.config.clustering.umap.n_components,
                    "embedding_weights": dict(self.config.clustering.embedding.weights)
                }
            }
            
            with open(f"{base_path}/metadata/training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Upload to blob storage
        try:
            blob_path = f"models/{tech_center_clean}/{year}/{quarter}"
            
            # Upload model files
            self.blob_storage.upload_file(
                f"{base_path}/clustering/hdbscan_clusterer.pkl",
                f"{blob_path}/clustering/hdbscan_clusterer.pkl"
            )
            
            self.blob_storage.upload_file(
                f"{base_path}/clustering/umap_reducer.pkl", 
                f"{blob_path}/clustering/umap_reducer.pkl"
            )
            
            # Upload analysis files
            self.blob_storage.upload_file(
                f"{base_path}/analysis/labeled_clusters.json",
                f"{blob_path}/analysis/labeled_clusters.json"
            )
            
            self.blob_storage.upload_file(
                f"{base_path}/analysis/domains.json",
                f"{blob_path}/analysis/domains.json"
            )
            
            # Upload metadata
            self.blob_storage.upload_file(
                f"{base_path}/metadata/training_metadata.json",
                f"{blob_path}/metadata/training_metadata.json"
            )
            
            logging.info(f"Model artifacts uploaded to blob storage: {blob_path}")
            
        except Exception as e:
            logging.error(f"Failed to upload artifacts to blob storage: {e}")
    
    def train_all_tech_centers_parallel(
        self, 
        year: Optional[int] = None, 
        quarter: Optional[str] = None
    ) -> Dict:
        """Train models for all tech centers in parallel"""
        
        # Use current quarter if not specified
        if year is None or quarter is None:
            year, quarter = self.get_current_quarter()
        
        start_time = time.time()
        logging.info(f"Starting parallel training for all tech centers - {year} {quarter}")
        
        results = []
        tech_centers = self.config.pipeline.tech_centers
        max_workers = min(self.config.pipeline.max_workers, len(tech_centers))
        
        if self.config.pipeline.parallel_training:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tech_center = {
                    executor.submit(self.train_tech_center_model, tc, year, quarter): tc 
                    for tc in tech_centers
                }
                
                for future in as_completed(future_to_tech_center):
                    tech_center = future_to_tech_center[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logging.info(f"Completed training for {tech_center}: {result['status']}")
                    except Exception as e:
                        logging.error(f"Training failed for {tech_center}: {e}")
                        results.append({
                            "tech_center": tech_center,
                            "year": year,
                            "quarter": quarter,
                            "status": "failed",
                            "error": str(e)
                        })
        else:
            # Sequential execution
            for tech_center in tech_centers:
                result = self.train_tech_center_model(tech_center, year, quarter)
                results.append(result)
        
        # Compile summary
        successful = len([r for r in results if r["status"] == "success"])
        failed = len([r for r in results if r["status"] == "failed"])
        skipped = len([r for r in results if r["status"] == "skipped"])
        
        summary = {
            "year": year,
            "quarter": quarter,
            "timestamp": datetime.now().isoformat(),
            "total_tech_centers": len(tech_centers),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "runtime_seconds": time.time() - start_time,
            "results": results
        }
        
        logging.info(f"Training completed: {successful} successful, {failed} failed, {skipped} skipped")
        
        # Save summary
        if self.config.pipeline.save_to_local:
            output_dir = f"{self.config.pipeline.result_path}/training/logs"
            os.makedirs(output_dir, exist_ok=True)
            
            with open(f"{output_dir}/training_run_{year}_{quarter}.json", "w") as f:
                json.dump(summary, f, indent=2)
        
        return summary
    
    def train_single_tech_center(
        self, 
        tech_center: str,
        year: Optional[int] = None,
        quarter: Optional[str] = None
    ) -> Dict:
        """Train model for a single tech center"""
        
        if year is None or quarter is None:
            year, quarter = self.get_current_quarter()
        
        return self.train_tech_center_model(tech_center, year, quarter)