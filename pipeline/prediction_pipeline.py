import logging
import time
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from data.bigquery_client import BigQueryClient
from data.blob_storage import BlobStorageClient
from preprocessing.text_processing import TextProcessor
from preprocessing.embedding_generation import EmbeddingGenerator

class PredictionPipeline:
    """
    Prediction pipeline that runs every 2 hours to:
    1. Load preprocessed data since last prediction
    2. Load trained models for each tech center
    3. Run predictions
    4. Store results with confidence scores
    """
    
    def __init__(self, config):
        self.config = config
        self.bigquery_client = BigQueryClient(config)
        self.blob_storage = BlobStorageClient(config)
        self.text_processor = TextProcessor(config)
        self.embedding_generator = EmbeddingGenerator(config)
        
        # Table names
        self.preprocessed_table = f"{config.bigquery.project_id}.{config.bigquery.datasets.preprocessing}.{config.bigquery.tables.preprocessed}"
        self.predictions_table = f"{config.bigquery.project_id}.{config.bigquery.datasets.results}.clustering_predictions"
        
        # Cache for loaded models
        self.model_cache = {}
    
    def get_prediction_watermark(self, tech_center: str) -> Optional[datetime]:
        """Get the last prediction timestamp for a tech center"""
        try:
            query = f"""
            SELECT MAX(processed_at) as last_predicted
            FROM `{self.predictions_table}`
            WHERE tech_center = '{tech_center}'
            """
            result = self.bigquery_client.run_query(query)
            if not result.empty and result.iloc[0]['last_predicted'] is not None:
                return result.iloc[0]['last_predicted']
            return None
        except Exception as e:
            logging.warning(f"Could not get prediction watermark for {tech_center}: {e}")
            return None
    
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
    
    def get_new_preprocessed_data(self, tech_center: str, watermark: Optional[datetime] = None) -> pd.DataFrame:
        """Get new preprocessed data for prediction"""
        
        # If no watermark, get last 2 hours
        if watermark is None:
            watermark = datetime.now() - timedelta(hours=2)
        
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
            AND processed_at > '{watermark}'
            AND processed_at <= CURRENT_TIMESTAMP()
        ORDER BY processed_at
        """
        
        logging.info(f"Getting new data for prediction: {tech_center} since {watermark}")
        df = self.bigquery_client.run_query(query)
        logging.info(f"Found {len(df)} new incidents for prediction: {tech_center}")
        
        return df
    
    def load_model_artifacts(self, tech_center: str, year: int, quarter: str) -> Optional[Dict]:
        """Load model artifacts for a tech center and quarter"""
        
        tech_center_clean = tech_center.replace(" ", "_").replace("-", "_")
        model_key = f"{tech_center_clean}_{year}_{quarter}"
        
        # Check cache first
        if model_key in self.model_cache:
            return self.model_cache[model_key]
        
        try:
            # Try to load from local storage first
            if self.config.pipeline.save_to_local:
                base_path = f"{self.config.pipeline.result_path}/models/{tech_center_clean}/{year}/{quarter}"
                
                if os.path.exists(f"{base_path}/clustering/hdbscan_clusterer.pkl"):
                    with open(f"{base_path}/clustering/hdbscan_clusterer.pkl", "rb") as f:
                        clusterer = pickle.load(f)
                    
                    with open(f"{base_path}/clustering/umap_reducer.pkl", "rb") as f:
                        reducer = pickle.load(f)
                    
                    with open(f"{base_path}/analysis/labeled_clusters.json", "r") as f:
                        labeled_clusters = json.load(f)
                    
                    with open(f"{base_path}/analysis/domains.json", "r") as f:
                        domains = json.load(f)
                    
                    artifacts = {
                        "clusterer": clusterer,
                        "reducer": reducer,
                        "labeled_clusters": labeled_clusters,
                        "domains": domains
                    }
                    
                    # Cache the artifacts
                    self.model_cache[model_key] = artifacts
                    logging.info(f"Loaded model artifacts from local storage: {model_key}")
                    return artifacts
            
            # If not found locally, try to download from blob storage
            blob_path = f"models/{tech_center_clean}/{year}/{quarter}"
            temp_dir = f"/tmp/models/{tech_center_clean}/{year}/{quarter}"
            os.makedirs(f"{temp_dir}/clustering", exist_ok=True)
            os.makedirs(f"{temp_dir}/analysis", exist_ok=True)
            
            # Download model files
            self.blob_storage.download_file(
                f"{blob_path}/clustering/hdbscan_clusterer.pkl",
                f"{temp_dir}/clustering/hdbscan_clusterer.pkl"
            )
            
            self.blob_storage.download_file(
                f"{blob_path}/clustering/umap_reducer.pkl",
                f"{temp_dir}/clustering/umap_reducer.pkl"
            )
            
            self.blob_storage.download_file(
                f"{blob_path}/analysis/labeled_clusters.json",
                f"{temp_dir}/analysis/labeled_clusters.json"
            )
            
            self.blob_storage.download_file(
                f"{blob_path}/analysis/domains.json",
                f"{temp_dir}/analysis/domains.json"
            )
            
            # Load artifacts
            with open(f"{temp_dir}/clustering/hdbscan_clusterer.pkl", "rb") as f:
                clusterer = pickle.load(f)
            
            with open(f"{temp_dir}/clustering/umap_reducer.pkl", "rb") as f:
                reducer = pickle.load(f)
            
            with open(f"{temp_dir}/analysis/labeled_clusters.json", "r") as f:
                labeled_clusters = json.load(f)
            
            with open(f"{temp_dir}/analysis/domains.json", "r") as f:
                domains = json.load(f)
            
            artifacts = {
                "clusterer": clusterer,
                "reducer": reducer,
                "labeled_clusters": labeled_clusters,
                "domains": domains
            }
            
            # Cache the artifacts
            self.model_cache[model_key] = artifacts
            logging.info(f"Downloaded and loaded model artifacts from blob storage: {model_key}")
            return artifacts
            
        except Exception as e:
            logging.error(f"Failed to load model artifacts for {tech_center} {year} {quarter}: {e}")
            return None
    
    def predict_incidents(self, df: pd.DataFrame, artifacts: Dict) -> pd.DataFrame:
        """Predict cluster assignments for new incidents"""
        
        if df.empty:
            return df
        
        try:
            clusterer = artifacts["clusterer"]
            reducer = artifacts["reducer"]
            labeled_clusters = artifacts["labeled_clusters"]
            domains = artifacts["domains"]
            
            # Extract embeddings
            embeddings = np.vstack(df['embedding'].values)
            
            # Apply UMAP transformation
            umap_embeddings = reducer.transform(embeddings)
            
            # Predict cluster labels and probabilities
            cluster_labels, cluster_probs = clusterer.approximate_predict(umap_embeddings)
            
            # Add predictions to dataframe
            df = df.copy()
            df['predicted_cluster'] = cluster_labels
            df['cluster_probability'] = cluster_probs
            
            # Map cluster labels to topics and domains
            df['cluster_topic'] = df['predicted_cluster'].astype(str).map(
                lambda x: labeled_clusters.get(x, {}).get("topic", "Unknown")
            )
            
            df['cluster_description'] = df['predicted_cluster'].astype(str).map(
                lambda x: labeled_clusters.get(x, {}).get("description", "No description")
            )
            
            # Map to domains
            cluster_to_domain = {}
            for domain in domains.get("domains", []):
                domain_name = domain.get("domain_name", "Unknown")
                for cluster_id in domain.get("clusters", []):
                    cluster_to_domain[cluster_id] = domain_name
            
            df['domain'] = df['predicted_cluster'].map(cluster_to_domain).fillna("Unknown")
            
            # Add prediction metadata
            df['predicted_at'] = datetime.now()
            df['model_version'] = "1.0"
            df['confidence_score'] = df['cluster_probability']
            
            # Flag low confidence predictions
            df['high_confidence'] = df['cluster_probability'] > 0.5
            
            return df
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return df
    
    def save_predictions(self, df: pd.DataFrame, tech_center: str):
        """Save predictions to BigQuery"""
        
        if df.empty:
            return
        
        try:
            # Select columns for storage
            prediction_df = df[[
                'number',
                'sys_created_on', 
                'combined_incidents_summary',
                'tech_center',
                'predicted_cluster',
                'cluster_probability',
                'cluster_topic',
                'cluster_description',
                'domain',
                'predicted_at',
                'model_version',
                'confidence_score',
                'high_confidence',
                'processed_at'
            ]].copy()
            
            success = self.bigquery_client.save_dataframe(
                prediction_df,
                self.predictions_table,
                write_disposition="WRITE_APPEND"
            )
            
            if success:
                logging.info(f"Successfully saved {len(prediction_df)} predictions for {tech_center}")
            else:
                logging.error(f"Failed to save predictions for {tech_center}")
                
        except Exception as e:
            logging.error(f"Error saving predictions for {tech_center}: {e}")
    
    def run_prediction_for_tech_center(self, tech_center: str) -> Dict:
        """Run prediction for a single tech center"""
        
        start_time = time.time()
        logging.info(f"Starting prediction for {tech_center}")
        
        try:
            # Get current quarter for model loading
            year, quarter = self.get_current_quarter()
            
            # Load model artifacts
            artifacts = self.load_model_artifacts(tech_center, year, quarter)
            if artifacts is None:
                return {
                    "tech_center": tech_center,
                    "status": "failed",
                    "reason": "Could not load model artifacts",
                    "runtime_seconds": time.time() - start_time
                }
            
            # Get watermark
            watermark = self.get_prediction_watermark(tech_center)
            
            # Get new data for prediction
            new_data = self.get_new_preprocessed_data(tech_center, watermark)
            
            if new_data.empty:
                logging.info(f"No new data for prediction: {tech_center}")
                return {
                    "tech_center": tech_center,
                    "status": "success",
                    "predicted_count": 0,
                    "runtime_seconds": time.time() - start_time
                }
            
            # Run predictions
            predictions = self.predict_incidents(new_data, artifacts)
            
            # Save predictions
            self.save_predictions(predictions, tech_center)
            
            # Calculate metrics
            high_confidence_count = len(predictions[predictions['high_confidence']])
            avg_confidence = predictions['confidence_score'].mean()
            
            runtime = time.time() - start_time
            logging.info(f"Prediction completed for {tech_center} in {runtime:.2f} seconds")
            
            return {
                "tech_center": tech_center,
                "status": "success",
                "predicted_count": len(predictions),
                "high_confidence_count": high_confidence_count,
                "avg_confidence": avg_confidence,
                "runtime_seconds": runtime
            }
            
        except Exception as e:
            logging.error(f"Prediction failed for {tech_center}: {e}")
            return {
                "tech_center": tech_center,
                "status": "failed",
                "error": str(e),
                "runtime_seconds": time.time() - start_time
            }
    
    def run_predictions_all_tech_centers(self) -> Dict:
        """Run predictions for all tech centers"""
        
        start_time = time.time()
        logging.info("Starting predictions for all tech centers")
        
        results = []
        total_predicted = 0
        total_high_confidence = 0
        
        for tech_center in self.config.pipeline.tech_centers:
            result = self.run_prediction_for_tech_center(tech_center)
            results.append(result)
            total_predicted += result.get("predicted_count", 0)
            total_high_confidence += result.get("high_confidence_count", 0)
        
        # Summary
        successful = len([r for r in results if r["status"] == "success"])
        failed = len([r for r in results if r["status"] == "failed"])
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tech_centers": len(self.config.pipeline.tech_centers),
            "successful": successful,
            "failed": failed,
            "total_predicted": total_predicted,
            "total_high_confidence": total_high_confidence,
            "runtime_seconds": time.time() - start_time,
            "results": results
        }
        
        logging.info(f"Predictions completed: {successful} successful, {failed} failed, {total_predicted} total predicted")
        
        # Save results to local storage for monitoring
        if self.config.pipeline.save_to_local:
            output_dir = f"{self.config.pipeline.result_path}/predictions/logs"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"{output_dir}/prediction_run_{timestamp}.json", "w") as f:
                json.dump(summary, f, indent=2)
        
        return summary