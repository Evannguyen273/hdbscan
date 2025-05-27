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
            default_summary_path = f"{intermediate_dir}/df_with_summaries.parquet"
            if os.path.exists(default_summary_path):
                logging.info(f"Found precomputed summaries at {default_summary_path}")
                summary_path = default_summary_path
        
        # Load data from BigQuery
        logging.info(f"Loading data with query: {input_query}")
        df = self.bigquery_client.run_query(input_query)
        logging.info(f"Loaded {len(df)} records")
        
        # Save raw data
        df.to_parquet(f"{output_dir}/raw_data.parquet", index=False)
        
        # Generate summaries if not using precomputed ones
        if summary_path and os.path.exists(summary_path):
            logging.info(f"Loading precomputed summaries from {summary_path}")
            summary_df = pd.read_parquet(summary_path)
            df['combined_incidents_summary'] = summary_df['combined_incidents_summary']
            fallback_stats = {"precomputed": True, "loaded_from": summary_path}
        else:
            logging.info("Processing text for embedding...")
            combined_summaries, fallback_stats = self.text_processor.process_incident_for_embedding_batch(
                df, batch_size=10
            )
            df['combined_incidents_summary'] = combined_summaries
            
            # Save intermediate dataframe
            summary_save_path = f"{intermediate_dir}/df_with_summaries.parquet"
            logging.info(f"Saving intermediate dataframe with summaries to {summary_save_path}")
            summary_df = df[['number', 'combined_incidents_summary']].copy()
            summary_df.to_parquet(summary_save_path, index=False)
        
        # Generate hybrid embeddings
        df_with_embeddings, classification_result, embedding_fallback_stats = self.embedding_generator.create_hybrid_embeddings(
            df, text_column='combined_incidents_summary'
        )
        
        # Merge fallback stats
        if "precomputed" not in fallback_stats:
            fallback_stats.update(embedding_fallback_stats)
        
        # Save embeddings
        df_with_embeddings.to_parquet(f"{output_dir}/df_with_embeddings.parquet", index=False)
        
        with open(f"{output_dir}/classification_result.json", "w") as f:
            json.dump(classification_result, f, indent=2)
        
        with open(f"{output_dir}/fallback_stats.json", "w") as f:
            json.dump(fallback_stats, f, indent=2)
        
        # Save to BigQuery if specified
        if embeddings_table_id:
            success = self.bigquery_client.save_dataframe(
                df_with_embeddings, embeddings_table_id, write_disposition
            )
            if success:
                logging.info(f"Successfully saved embeddings to {embeddings_table_id}")
        
        # Save metadata
        total_time = time.time() - start_time
        metadata = {
            "start_time": datetime.now().isoformat(),
            "record_count": len(df),
            "dataset_name": dataset_name,
            "using_precomputed_summaries": summary_path is not None,
            "runtime_seconds": total_time
        }
        
        with open(f"{output_dir}/embedding_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Embeddings generation completed in {total_time:.2f} seconds")
        return df_with_embeddings, classification_result, fallback_stats
    
    def stage_2_train_hdbscan(
        self,
        df_with_embeddings: Optional[pd.DataFrame] = None,
        dataset_name: str = "",
        embedding_path: Optional[str] = None,
        use_checkpoint: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray, Any, Any]:
        """Stage 2: Train HDBSCAN clustering model"""
        
        logging.info("=== STAGE 2: TRAINING HDBSCAN ===")
        
        # Load embeddings if not provided
        if df_with_embeddings is None:
            if embedding_path:
                if not os.path.exists(embedding_path):
                    raise FileNotFoundError(f"Embeddings file not found at {embedding_path}")
                logging.info(f"Loading embeddings from custom path: {embedding_path}")
                df_with_embeddings = pd.read_parquet(embedding_path)
            elif dataset_name:
                default_path = f"{self.config.pipeline.result_path}/{dataset_name}/embeddings/df_with_embeddings.parquet"
                if not os.path.exists(default_path):
                    raise FileNotFoundError(f"Embeddings file not found at {default_path}")
                logging.info(f"Loading embeddings from default path: {default_path}")
                df_with_embeddings = pd.read_parquet(default_path)
            else:
                raise ValueError("Either df_with_embeddings, dataset_name, or embedding_path must be provided")
        
        # Train HDBSCAN
        clustered_df, umap_embeddings, clusterer, reducer = self.clusterer.train_hdbscan(
            df_with_embeddings=df_with_embeddings,
            dataset_name=dataset_name,
            result_path=self.config.pipeline.result_path,
            use_checkpoint=use_checkpoint
        )
        
        return clustered_df, umap_embeddings, clusterer, reducer
    
    def stage_3_analyze_clusters(
        self,
        clustered_df: Optional[pd.DataFrame] = None,
        dataset_name: str = "",
        umap_embeddings: Optional[np.ndarray] = None,
        cluster_labels: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
        """Stage 3: Generate cluster info, label clusters and group into domains"""
        
        start_time = time.time()
        logging.info("=== STAGE 3: ANALYZING CLUSTERS ===")
        
        # Load clustered data if not provided
        if clustered_df is None:
            if not dataset_name:
                raise ValueError("Either clustered_df or dataset_name must be provided")
            
            cluster_path = f"{self.config.pipeline.result_path}/{dataset_name}/clustering/clustered_df.parquet"
            if not os.path.exists(cluster_path):
                raise FileNotFoundError(f"Clustered data not found at {cluster_path}")
            
            logging.info(f"Loading clustered data from {cluster_path}")
            essential_cols = ['number', 'short_description', 'cluster', 'cluster_probability', 'embedding']
            clustered_df = read_parquet_optimized(cluster_path, columns=essential_cols)
        
        # Create output directory
        output_dir = f"{self.config.pipeline.result_path}/{dataset_name}/analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate cluster information
        clusters_info = self.cluster_analyzer.generate_cluster_info(
            clustered_df,
            text_column='short_description',
            cluster_column='cluster',
            sample_size=5,
            output_dir=output_dir
        )
        
        # Label clusters using LLM
        try:
            labeled_clusters = self.cluster_labeler.label_clusters_with_llm(
                clusters_info,
                max_samples=300,
                chunk_size=25,
                output_dir=output_dir
            )
        except Exception as e:
            logging.error(f"Error in cluster labeling: {e}")
            # Create fallback labels
            labeled_clusters = {
                str(cid): {"topic": f"Cluster {cid}", "description": "Auto-generated label"}
                for cid in clustered_df['cluster'].unique() if cid != -1
            }
            labeled_clusters["-1"] = {"topic": "Noise", "description": "Unclustered data points"}
            
            with open(f"{output_dir}/labeled_clusters_fallback.json", "w") as f:
                json.dump(labeled_clusters, f, indent=2)
        
        # Group clusters into domains
        try:
            # Load UMAP embeddings if not provided
            if umap_embeddings is None:
                umap_embeddings_path = f"{self.config.pipeline.result_path}/{dataset_name}/clustering/umap_embeddings.npy"
                umap_embeddings = np.load(umap_embeddings_path)
            
            if cluster_labels is None:
                cluster_labels = clustered_df['cluster'].values
            
            domains = self.domain_grouper.group_clusters_into_domains(
                labeled_clusters,
                clusters_info,
                umap_embeddings,
                cluster_labels,
                output_dir=output_dir
            )
        except Exception as e:
            logging.error(f"Error in domain grouping: {e}")
            # Create fallback domains
            domains = {
                "domains": [
                    {"domain_name": "Other", "clusters": list(map(int, clusters_info.keys()))},
                    {"domain_name": "Noise", "description": "Uncategorized incidents", "clusters": [-1]}
                ]
            }
            
            with open(f"{output_dir}/domains_fallback.json", "w") as f:
                json.dump(domains, f, indent=2)
        
        # Apply labels to data
        final_df = self.cluster_analyzer.apply_labels_to_data(clustered_df, labeled_clusters, domains)
        
        # Save final dataframe
        final_df.to_parquet(f"{output_dir}/final_df.parquet", index=False)
        
        # Save metadata
        total_time = time.time() - start_time
        with open(f"{output_dir}/analysis_metadata.json", "w") as f:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "max_domains": self.config.clustering.max_domains
                },
                "results": {
                    "num_clusters": len([k for k in clusters_info.keys() if k != "-1"]),
                    "num_domains": len(domains.get("domains", [])),
                    "noise_percentage": clusters_info.get("-1", {}).get("percentage", 0) if "-1" in clusters_info else 0
                },
                "runtime_seconds": total_time,
                "dataset_name": dataset_name
            }
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Cluster analysis completed in {total_time:.2f} seconds")
        return final_df, clusters_info, labeled_clusters, domains
    
    def stage_4_save_results(
        self,
        final_df: Optional[pd.DataFrame] = None,
        dataset_name: str = "",
        results_table_id: Optional[str] = None,
        write_disposition: str = "WRITE_APPEND"
    ) -> bool:
        """Stage 4: Save results to BigQuery"""
        
        start_time = time.time()
        logging.info("=== STAGE 4: SAVING RESULTS ===")
        
        # Load final dataframe if not provided
        if final_df is None:
            if not dataset_name:
                raise ValueError("Either final_df or dataset_name must be provided")
            
            final_path = f"{self.config.pipeline.result_path}/{dataset_name}/analysis/final_df.parquet"
            if not os.path.exists(final_path):
                raise FileNotFoundError(f"Final data not found at {final_path}")
            
            logging.info(f"Loading final data from {final_path}")
            final_df = read_parquet_optimized(final_path)
        
        # Create output directory
        output_dir = f"{self.config.pipeline.result_path}/{dataset_name}/final"
        os.makedirs(output_dir, exist_ok=True)
        
        success = False
        
        # Save to BigQuery if table_id provided
        if results_table_id:
            try:
                success = self.bigquery_client.save_dataframe(
                    final_df, results_table_id, write_disposition
                )
                if success:
                    logging.info(f"Successfully saved results to BigQuery table {results_table_id}")
                else:
                    logging.error("Failed to save results to BigQuery")
            except Exception as e:
                logging.error(f"Error saving to BigQuery: {e}")
                success = False
        else:
            logging.info("No BigQuery table provided, skipping BigQuery save")
        
        # Save final CSV locally
        final_df.to_csv(f"{output_dir}/results.csv", index=False)
        
        total_time = time.time() - start_time
        logging.info(f"Results saving completed in {total_time:.2f} seconds")
        
        return success
    
    def run_modular_pipeline(
        self,
        input_query: str,
        embeddings_table_id: Optional[str],
        results_table_id: Optional[str],
        dataset_name: str,
        embedding_path: Optional[str] = None,
        summary_path: Optional[str] = None,
        write_disposition: str = "WRITE_APPEND",
        start_from_stage: int = 1,
        end_at_stage: int = 4,
        use_checkpoint: bool = True
    ) -> Dict:
        """Run the complete modular clustering pipeline"""
        
        # Validate stages
        if not 1 <= start_from_stage <= 4:
            raise ValueError("start_from_stage must be between 1 and 4")
        if not 1 <= end_at_stage <= 4:
            raise ValueError("end_at_stage must be between 1 and 4")
        if start_from_stage > end_at_stage:
            raise ValueError("start_from_stage cannot be greater than end_at_stage")
        
        start_time = time.time()
        logging.info(f"Starting modular pipeline for dataset '{dataset_name}' (stages {start_from_stage}-{end_at_stage})")
        
        results = {}
        
        # Stage 1: Generate embeddings
        if start_from_stage <= 1 <= end_at_stage:
            df_with_embeddings, classification_result, fallback_stats = self.stage_1_generate_embeddings(
                input_query=input_query,
                dataset_name=dataset_name,
                embeddings_table_id=embeddings_table_id,
                summary_path=summary_path,
                write_disposition=write_disposition
            )
            results["embeddings"] = {
                "df_with_embeddings": df_with_embeddings,
                "classification_result": classification_result,
                "fallback_stats": fallback_stats
            }
        
        # Stage 2: Train HDBSCAN
        if start_from_stage <= 2 <= end_at_stage:
            df_input = results.get("embeddings", {}).get("df_with_embeddings", None)
            
            clustered_df, umap_embeddings, clusterer, reducer = self.stage_2_train_hdbscan(
                df_with_embeddings=df_input,
                dataset_name=dataset_name,
                embedding_path=embedding_path,
                use_checkpoint=use_checkpoint
            )
            results["clustering"] = {
                "clustered_df": clustered_df,
                "umap_embeddings": umap_embeddings,
                "clusterer": clusterer,
                "reducer": reducer
            }
        
        # Stage 3: Analyze clusters
        if start_from_stage <= 3 <= end_at_stage:
            df_input = results.get("clustering", {}).get("clustered_df", None)
            umap_input = results.get("clustering", {}).get("umap_embeddings", None)
            cluster_labels_input = None
            if df_input is not None:
                cluster_labels_input = df_input['cluster'].values
            
            final_df, clusters_info, labeled_clusters, domains = self.stage_3_analyze_clusters(
                clustered_df=df_input,
                dataset_name=dataset_name,
                umap_embeddings=umap_input,
                cluster_labels=cluster_labels_input
            )
            results["analysis"] = {
                "final_df": final_df,
                "clusters_info": clusters_info,
                "labeled_clusters": labeled_clusters,
                "domains": domains
            }
        
        # Stage 4: Save results
        if start_from_stage <= 4 <= end_at_stage:
            df_input = results.get("analysis", {}).get("final_df", None)
            
            success = self.stage_4_save_results(
                final_df=df_input,
                dataset_name=dataset_name,
                results_table_id=results_table_id,
                write_disposition=write_disposition
            )
            results["saved"] = success
        
        # Calculate total runtime and save metadata
        total_time = time.time() - start_time
        logging.info(f"Modular pipeline completed in {total_time:.2f} seconds")
        
        # Save consolidated metadata
        output_dir = f"{self.config.pipeline.result_path}/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/run_metadata.json", "w") as f:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "dataset_name": dataset_name,
                "stages_run": list(range(start_from_stage, end_at_stage + 1)),
                "parameters": {
                    "min_cluster_size": self.config.clustering.min_cluster_size,
                    "min_samples": self.config.clustering.min_samples,
                    "umap_n_components": self.config.clustering.umap_n_components,
                    "max_domains": self.config.clustering.max_domains,
                    "write_disposition": write_disposition,
                    "use_checkpoint": use_checkpoint
                },
                "runtime_seconds": total_time
            }
            json.dump(metadata, f, indent=2)
        
        return results