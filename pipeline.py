import logging
import time
import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np

from config.config import load_config
from data.bigquery_client import BigQueryClient
from data.blob_storage import BlobStorageClient
from preprocessing.text_processing import TextProcessor
from preprocessing.embedding_generation import EmbeddingGenerator
from core.clustering import HDBSCANClusterer
from analysis.cluster_analysis import ClusterAnalyzer
from analysis.cluster_labeling import ClusterLabeler
from analysis.domain_grouping import DomainGrouper
from utils.file_utils import read_parquet_optimized

class ClusteringPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the clustering pipeline with configuration"""
        self.config = load_config(config_path)
        
        # Initialize clients and components
        self.bigquery_client = BigQueryClient(self.config)
        self.blob_storage = BlobStorageClient(self.config)
        self.text_processor = TextProcessor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.clusterer = HDBSCANClusterer(self.config)
        self.cluster_analyzer = ClusterAnalyzer(self.config)
        self.cluster_labeler = ClusterLabeler(self.config)
        self.domain_grouper = DomainGrouper(self.config)
    
    def stage_1_generate_embeddings(
        self,
        input_query: str,
        dataset_name: str,
        embeddings_table_id: Optional[str] = None,
        summary_path: Optional[str] = None,
        write_disposition: str = "WRITE_APPEND"
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Stage 1: Generate hybrid embeddings from text data"""
        
        start_time = time.time()
        logging.info("=== STAGE 1: GENERATING EMBEDDINGS ===")
        
        # Create output directories
        output_dir = f"{self.config.pipeline.result_path}/{dataset_name}/embeddings"
        intermediate_dir = f"{self.config.pipeline.result_path}/{dataset_name}/intermediate"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Check for precomputed summaries
        if not summary_path:
            # Load data from BigQuery
            logging.info(f"Loading data with query: {input_query}")
            df = self.bigquery_client.run_query(input_query)
            logging.info(f"Loaded {len(df)} records")
            
            # Save raw data
            df.to_parquet(f"{output_dir}/raw_data.parquet", index=False)
            
            # Generate summaries
            logging.info("Generating text summaries...")
            df_with_summaries = self.text_processor.process_dataframe(df)
            
            # Save intermediate summaries
            df_with_summaries.to_parquet(f"{intermediate_dir}/df_with_summaries.parquet", index=False)
        else:
            # Load precomputed summaries
            logging.info(f"Loading precomputed summaries from {summary_path}")
            df_with_summaries = pd.read_parquet(summary_path)
        
        # Generate embeddings
        logging.info("Generating embeddings...")
        df_with_embeddings = self.embedding_generator.generate_embeddings(df_with_summaries)
        
        # Save results
        df_with_embeddings.to_parquet(f"{output_dir}/df_with_embeddings.parquet", index=False)
        
        # Calculate embedding metadata
        embedding_metadata = {
            "total_records": len(df_with_embeddings),
            "embedding_dimension": len(df_with_embeddings['embedding'].iloc[0]) if len(df_with_embeddings) > 0 else 0,
            "generation_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "weights": {
                "entity": self.config.clustering.embedding.weights.entity,
                "action": self.config.clustering.embedding.weights.action,
                "semantic": self.config.clustering.embedding.weights.semantic
            }
        }
        
        # Save metadata
        with open(f"{output_dir}/embedding_metadata.json", 'w') as f:
            json.dump(embedding_metadata, f, indent=2)
        
        # Upload to BigQuery if specified
        if embeddings_table_id:
            logging.info(f"Uploading embeddings to BigQuery table: {embeddings_table_id}")
            self.bigquery_client.upload_dataframe(
                df_with_embeddings, 
                embeddings_table_id, 
                write_disposition=write_disposition
            )
        
        stage_1_results = {
            "dataframe": df_with_embeddings,
            "metadata": embedding_metadata,
            "output_dir": output_dir
        }
        
        logging.info(f"Stage 1 completed in {embedding_metadata['generation_time']:.2f} seconds")
        return df_with_embeddings, embedding_metadata, stage_1_results
    
    def stage_2_train_clustering(
        self,
        df_with_embeddings: pd.DataFrame,
        dataset_name: str
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Stage 2: Train UMAP + HDBSCAN clustering"""
        
        start_time = time.time()
        logging.info("=== STAGE 2: TRAINING CLUSTERING ===")
        
        # Create output directory
        output_dir = f"{self.config.pipeline.result_path}/{dataset_name}/clustering"
        os.makedirs(output_dir, exist_ok=True)
        
        # Train clustering
        clustered_df = self.clusterer.fit_predict(df_with_embeddings)
        
        # Save clustering results
        clustered_df.to_parquet(f"{output_dir}/clustered_df.parquet", index=False)
        
        # Save clustering artifacts
        self.clusterer.save_artifacts(output_dir)
        
        # Calculate clustering metadata
        clustering_metadata = {
            "total_records": len(clustered_df),
            "n_clusters": clustered_df['cluster'].nunique() - (1 if -1 in clustered_df['cluster'].values else 0),
            "n_noise": (clustered_df['cluster'] == -1).sum(),
            "training_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "min_cluster_size": self.config.clustering.hdbscan.min_cluster_size,
                "min_samples": self.config.clustering.hdbscan.min_samples,
                "umap_n_components": self.config.clustering.umap.n_components,
                "umap_n_neighbors": self.config.clustering.umap.n_neighbors
            }
        }
        
        # Save metadata
        with open(f"{output_dir}/clustering_metadata.json", 'w') as f:
            json.dump(clustering_metadata, f, indent=2)
        
        stage_2_results = {
            "dataframe": clustered_df,
            "metadata": clustering_metadata,
            "output_dir": output_dir
        }
        
        logging.info(f"Stage 2 completed in {clustering_metadata['training_time']:.2f} seconds")
        logging.info(f"Found {clustering_metadata['n_clusters']} clusters with {clustering_metadata['n_noise']} noise points")
        
        return clustered_df, clustering_metadata, stage_2_results
    
    def stage_3_analyze_clusters(
        self,
        clustered_df: pd.DataFrame,
        dataset_name: str
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Stage 3: Analyze clusters and generate labels"""
        
        start_time = time.time()
        logging.info("=== STAGE 3: ANALYZING CLUSTERS ===")
        
        # Create output directory
        output_dir = f"{self.config.pipeline.result_path}/{dataset_name}/analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze clusters
        cluster_details = self.cluster_analyzer.analyze_clusters(clustered_df)
        
        # Generate cluster labels
        labeled_clusters = self.cluster_labeler.label_clusters(clustered_df, cluster_details)
        
        # Group into domains
        domains = self.domain_grouper.group_clusters(labeled_clusters)
        
        # Apply labels to dataframe
        final_df = clustered_df.copy()
        
        # Add cluster topics
        cluster_topic_map = {cluster['cluster_id']: cluster['topic'] for cluster in labeled_clusters}
        final_df['cluster_topic'] = final_df['cluster'].map(cluster_topic_map).fillna('Unknown')
        
        # Add domain information
        cluster_domain_map = {}
        for domain_name, domain_info in domains.items():
            for cluster_id in domain_info['clusters']:
                cluster_domain_map[cluster_id] = domain_name
        final_df['domain_name'] = final_df['cluster'].map(cluster_domain_map).fillna('Uncategorized')
        
        # Save results
        final_df.to_parquet(f"{output_dir}/final_df.parquet", index=False)
        
        # Save analysis artifacts
        with open(f"{output_dir}/cluster_details.json", 'w') as f:
            json.dump(cluster_details, f, indent=2)
        
        with open(f"{output_dir}/labeled_clusters.json", 'w') as f:
            json.dump(labeled_clusters, f, indent=2)
        
        with open(f"{output_dir}/domains.json", 'w') as f:
            json.dump(domains, f, indent=2)
        
        # Calculate analysis metadata
        analysis_metadata = {
            "total_records": len(final_df),
            "labeled_clusters": len(labeled_clusters),
            "domains": len(domains),
            "analysis_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save metadata
        with open(f"{output_dir}/analysis_metadata.json", 'w') as f:
            json.dump(analysis_metadata, f, indent=2)
        
        stage_3_results = {
            "dataframe": final_df,
            "metadata": analysis_metadata,
            "cluster_details": cluster_details,
            "labeled_clusters": labeled_clusters,
            "domains": domains,
            "output_dir": output_dir
        }
        
        logging.info(f"Stage 3 completed in {analysis_metadata['analysis_time']:.2f} seconds")
        logging.info(f"Generated {len(labeled_clusters)} cluster labels in {len(domains)} domains")
        
        return final_df, analysis_metadata, stage_3_results
    
    def stage_4_save_results(
        self,
        final_df: pd.DataFrame,
        dataset_name: str,
        results_table_id: Optional[str] = None,
        write_disposition: str = "WRITE_APPEND"
    ) -> Dict:
        """Stage 4: Save final results"""
        
        start_time = time.time()
        logging.info("=== STAGE 4: SAVING RESULTS ===")
        
        # Create output directory
        output_dir = f"{self.config.pipeline.result_path}/{dataset_name}/final"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV for easy viewing
        results_csv_path = f"{output_dir}/results.csv"
        final_df.to_csv(results_csv_path, index=False)
        logging.info(f"Results saved to {results_csv_path}")
        
        # Upload to BigQuery if specified
        if results_table_id:
            logging.info(f"Uploading results to BigQuery table: {results_table_id}")
            self.bigquery_client.upload_dataframe(
                final_df,
                results_table_id,
                write_disposition=write_disposition
            )
        
        # Upload to blob storage if configured
        if self.config.azure.blob_connection_string:
            blob_path = f"{dataset_name}/final/results.parquet"
            final_df.to_parquet("temp_results.parquet", index=False)
            success = self.blob_storage.upload_file("temp_results.parquet", blob_path)
            if success:
                logging.info(f"Results uploaded to blob storage: {blob_path}")
            os.remove("temp_results.parquet")
        
        stage_4_results = {
            "csv_path": results_csv_path,
            "total_records": len(final_df),
            "save_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        logging.info(f"Stage 4 completed in {stage_4_results['save_time']:.2f} seconds")
        return stage_4_results
    
    def run_modular_pipeline(
        self,
        input_query: str,
        embeddings_table_id: Optional[str] = None,
        results_table_id: Optional[str] = None,
        dataset_name: str = "default",
        embedding_path: Optional[str] = None,
        summary_path: Optional[str] = None,
        write_disposition: str = "WRITE_APPEND",
        start_from_stage: int = 1,
        end_at_stage: int = 4,
        use_checkpoint: bool = True
    ) -> Dict:
        """Run the complete modular pipeline with checkpointing"""
        
        pipeline_start = time.time()
        logging.info(f"Starting modular pipeline for dataset: {dataset_name}")
        logging.info(f"Stages: {start_from_stage} to {end_at_stage}")
        
        # Initialize run metadata
        run_metadata = {
            "dataset_name": dataset_name,
            "start_stage": start_from_stage,
            "end_stage": end_at_stage,
            "start_time": datetime.now().isoformat(),
            "stages": {}
        }
        
        # Stage variables
        df_with_embeddings = None
        clustered_df = None
        final_df = None
        
        try:
            # Stage 1: Generate embeddings
            if start_from_stage <= 1 <= end_at_stage:
                if use_checkpoint and embedding_path and os.path.exists(embedding_path):
                    logging.info(f"Loading embeddings from checkpoint: {embedding_path}")
                    df_with_embeddings = read_parquet_optimized(embedding_path)
                    stage_1_metadata = {"loaded_from_checkpoint": True}
                else:
                    df_with_embeddings, stage_1_metadata, stage_1_results = self.stage_1_generate_embeddings(
                        input_query, dataset_name, embeddings_table_id, summary_path, write_disposition
                    )
                
                run_metadata["stages"]["stage_1"] = stage_1_metadata
            
            # Stage 2: Train clustering
            if start_from_stage <= 2 <= end_at_stage:
                if df_with_embeddings is None:
                    # Load from checkpoint
                    checkpoint_path = f"{self.config.pipeline.result_path}/{dataset_name}/embeddings/df_with_embeddings.parquet"
                    if os.path.exists(checkpoint_path):
                        logging.info(f"Loading embeddings from checkpoint: {checkpoint_path}")
                        df_with_embeddings = read_parquet_optimized(checkpoint_path)
                    else:
                        raise ValueError("No embeddings available. Run stage 1 first.")
                
                clustered_df, stage_2_metadata, stage_2_results = self.stage_2_train_clustering(
                    df_with_embeddings, dataset_name
                )
                run_metadata["stages"]["stage_2"] = stage_2_metadata
            
            # Stage 3: Analyze clusters
            if start_from_stage <= 3 <= end_at_stage:
                if clustered_df is None:
                    # Load from checkpoint
                    checkpoint_path = f"{self.config.pipeline.result_path}/{dataset_name}/clustering/clustered_df.parquet"
                    if os.path.exists(checkpoint_path):
                        logging.info(f"Loading clustered data from checkpoint: {checkpoint_path}")
                        clustered_df = read_parquet_optimized(checkpoint_path)
                    else:
                        raise ValueError("No clustered data available. Run stage 2 first.")
                
                final_df, stage_3_metadata, stage_3_results = self.stage_3_analyze_clusters(
                    clustered_df, dataset_name
                )
                run_metadata["stages"]["stage_3"] = stage_3_metadata
            
            # Stage 4: Save results
            if start_from_stage <= 4 <= end_at_stage:
                if final_df is None:
                    # Load from checkpoint
                    checkpoint_path = f"{self.config.pipeline.result_path}/{dataset_name}/analysis/final_df.parquet"
                    if os.path.exists(checkpoint_path):
                        logging.info(f"Loading final data from checkpoint: {checkpoint_path}")
                        final_df = read_parquet_optimized(checkpoint_path)
                    else:
                        raise ValueError("No final data available. Run stage 3 first.")
                
                stage_4_results = self.stage_4_save_results(
                    final_df, dataset_name, results_table_id, write_disposition
                )
                run_metadata["stages"]["stage_4"] = stage_4_results
            
            # Complete run metadata
            run_metadata["end_time"] = datetime.now().isoformat()
            run_metadata["total_time"] = time.time() - pipeline_start
            run_metadata["status"] = "completed"
            
            # Save run metadata
            metadata_dir = f"{self.config.pipeline.result_path}/{dataset_name}"
            os.makedirs(metadata_dir, exist_ok=True)
            with open(f"{metadata_dir}/run_metadata.json", 'w') as f:
                json.dump(run_metadata, f, indent=2)
            
            logging.info(f"Pipeline completed successfully in {run_metadata['total_time']:.2f} seconds")
            return run_metadata
        
        except Exception as e:
            run_metadata["end_time"] = datetime.now().isoformat()
            run_metadata["total_time"] = time.time() - pipeline_start
            run_metadata["status"] = "failed"
            run_metadata["error"] = str(e)
            
            logging.error(f"Pipeline failed: {e}")
            raise e