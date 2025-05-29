"""
HDBSCAN Prediction Pipeline
Handles real-time incident classification using trained models
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

from data.bigquery_client import BigQueryClient
from data.blob_storage import BlobStorageClient
from preprocessing.embedding_generation import EmbeddingGenerator
from preprocessing.text_processing import TextProcessor
from utils.error_handler import PipelineLogger, catch_errors
from config.config import load_config


class PredictionPipeline:
    """
    Prediction pipeline for classifying new incidents using trained HDBSCAN models
    Runs every 2 hours to process new incidents
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = PipelineLogger("prediction")
        
        # Initialize clients
        self.bigquery_client = BigQueryClient(config)
        self.blob_client = BlobStorageClient(config)
        
        # Initialize processors
        self.text_processor = TextProcessor(config)
        self.embedding_generator = EmbeddingGenerator(config)
        
        # Model cache
        self.loaded_models = {}
        
        logging.info("PredictionPipeline initialized")
    
    @catch_errors
    def run_predictions_all_tech_centers(self) -> Dict[str, Any]:
        """
        Run predictions for all tech centers
        Returns summary of prediction results
        """
        self.logger.log_stage_start("predictions_all_tech_centers", {
            "tech_centers_count": len(self.config.pipeline.tech_centers)
        })
        
        results = {
            "total_predicted": 0,
            "successful_centers": 0,
            "failed_centers": 0,
            "tech_center_results": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for i, tech_center in enumerate(self.config.pipeline.tech_centers):
            try:
                self.logger.log_progress(i + 1, len(self.config.pipeline.tech_centers), "tech centers")
                
                # Run prediction for individual tech center
                center_result = self.run_prediction_for_tech_center(tech_center)
                
                results["tech_center_results"][tech_center] = center_result
                results["total_predicted"] += center_result.get("predicted_count", 0)
                results["successful_centers"] += 1
                
                self.logger.log_stage_complete(f"prediction_{tech_center}", center_result)
                
            except Exception as e:
                self.logger.log_error("tech_center_prediction", e, {"tech_center": tech_center})
                results["failed_centers"] += 1
                results["tech_center_results"][tech_center] = {
                    "status": "failed",
                    "error": str(e),
                    "predicted_count": 0
                }
        
        self.logger.log_stage_complete("predictions_all_tech_centers", results)
        return results
    
    @catch_errors
    def run_prediction_for_tech_center(self, tech_center: str) -> Dict[str, Any]:
        """
        Run predictions for a specific tech center
        """
        self.logger.log_stage_start(f"prediction_{tech_center}", {"tech_center": tech_center})
        
        # Get latest model info
        model_info = self._get_latest_model_info(tech_center)
        if not model_info:
            raise ValueError(f"No trained model found for tech center: {tech_center}")
        
        # Load trained models
        models = self._load_models_for_tech_center(tech_center, model_info)
        
        # Get new incidents since last prediction
        new_incidents = self._get_new_incidents_for_tech_center(tech_center)
        
        if len(new_incidents) == 0:
            self.logger.log_stage_complete(f"prediction_{tech_center}", {
                "predicted_count": 0,
                "status": "no_new_incidents"
            })
            return {
                "predicted_count": 0,
                "status": "no_new_incidents",
                "tech_center": tech_center
            }
        
        # Process and predict
        predictions = self._predict_incidents(new_incidents, models, tech_center)
        
        # Store predictions
        self._store_predictions(predictions, tech_center)
        
        # Update watermark
        self._update_prediction_watermark(tech_center, new_incidents)
        
        result = {
            "predicted_count": len(predictions),
            "status": "completed",
            "tech_center": tech_center,
            "model_quarter": model_info.get("quarter"),
            "model_year": model_info.get("year"),
            "cluster_distribution": self._get_cluster_distribution(predictions)
        }
        
        self.logger.log_stage_complete(f"prediction_{tech_center}", result)
        return result
    
    def _get_latest_model_info(self, tech_center: str) -> Optional[Dict[str, Any]]:
        """
        Get information about the latest trained model for tech center
        """
        try:
            # Query BigQuery for latest model metadata
            query = f"""
            SELECT 
                tech_center,
                year,
                quarter,
                model_path,
                training_timestamp,
                model_metrics
            FROM `{self.config.bigquery.project_id}.{self.config.bigquery.datasets.results}.model_metadata`
            WHERE tech_center = @tech_center
            ORDER BY training_timestamp DESC
            LIMIT 1
            """
            
            job_config = self.bigquery_client.create_query_job_config()
            job_config.query_parameters = [
                self.bigquery_client.bigquery.ScalarQueryParameter("tech_center", "STRING", tech_center)
            ]
            
            result = self.bigquery_client.execute_query(query, job_config)
            
            if len(result) == 0:
                return None
            
            return result.iloc[0].to_dict()
            
        except Exception as e:
            logging.error(f"Failed to get model info for {tech_center}: {e}")
            return None
    
    def _load_models_for_tech_center(self, tech_center: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load trained models (HDBSCAN, UMAP, etc.) for tech center
        """
        cache_key = f"{tech_center}_{model_info['year']}_{model_info['quarter']}"
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        try:
            # Build model paths
            year = model_info["year"]
            quarter = model_info["quarter"]
            base_path = f"models/{tech_center}/{year}/{quarter}"
            
            models = {}
            
            # Load HDBSCAN model
            hdbscan_path = f"{base_path}/hdbscan_model.pkl"
            models["hdbscan"] = self._load_model_from_blob(hdbscan_path)
            
            # Load UMAP model
            umap_path = f"{base_path}/umap_model.pkl"
            models["umap"] = self._load_model_from_blob(umap_path)
            
            # Load preprocessing artifacts
            preprocessing_path = f"{base_path}/preprocessing_artifacts.pkl"
            models["preprocessing"] = self._load_model_from_blob(preprocessing_path)
            
            # Load cluster metadata
            metadata_path = f"{base_path}/cluster_metadata.json"
            models["metadata"] = self.blob_client.download_json(metadata_path)
            
            # Cache models
            self.loaded_models[cache_key] = models
            
            logging.info(f"Loaded models for {tech_center} ({year} {quarter})")
            return models
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models for {tech_center}: {e}")
    
    def _load_model_from_blob(self, blob_path: str) -> Any:
        """
        Load a pickled model from blob storage
        """
        try:
            model_bytes = self.blob_client.download_blob(blob_path)
            return pickle.loads(model_bytes)
        except Exception as e:
            # Try joblib if pickle fails
            try:
                model_bytes = self.blob_client.download_blob(blob_path)
                return joblib.loads(model_bytes)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model from {blob_path}: {e}, {e2}")
    
    def _get_new_incidents_for_tech_center(self, tech_center: str) -> pd.DataFrame:
        """
        Get new incidents for tech center since last prediction
        """
        try:
            # Get last prediction timestamp (watermark)
            last_prediction_time = self._get_prediction_watermark(tech_center)
            
            # Query for new incidents
            query = f"""
            SELECT 
                incident_id,
                description,
                short_description,
                tech_center,
                sys_created_on,
                category,
                subcategory,
                assignment_group
            FROM `{self.config.bigquery.project_id}.{self.config.bigquery.datasets.preprocessing}.preprocessed_incidents`
            WHERE tech_center = @tech_center
            AND sys_created_on > @last_prediction_time
            ORDER BY sys_created_on ASC
            """
            
            job_config = self.bigquery_client.create_query_job_config()
            job_config.query_parameters = [
                self.bigquery_client.bigquery.ScalarQueryParameter("tech_center", "STRING", tech_center),
                self.bigquery_client.bigquery.ScalarQueryParameter("last_prediction_time", "TIMESTAMP", last_prediction_time)
            ]
            
            result = self.bigquery_client.execute_query(query, job_config)
            
            logging.info(f"Retrieved {len(result)} new incidents for {tech_center}")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get new incidents for {tech_center}: {e}")
    
    def _get_prediction_watermark(self, tech_center: str) -> datetime:
        """
        Get the timestamp of last prediction for tech center
        """
        try:
            query = f"""
            SELECT MAX(prediction_timestamp) as last_prediction
            FROM `{self.config.bigquery.project_id}.{self.config.bigquery.datasets.results}.prediction_watermarks`
            WHERE tech_center = @tech_center
            """
            
            job_config = self.bigquery_client.create_query_job_config()
            job_config.query_parameters = [
                self.bigquery_client.bigquery.ScalarQueryParameter("tech_center", "STRING", tech_center)
            ]
            
            result = self.bigquery_client.execute_query(query, job_config)
            
            if len(result) > 0 and result.iloc[0]["last_prediction"] is not None:
                return result.iloc[0]["last_prediction"]
            else:
                # Default to 2 hours ago if no watermark exists
                return datetime.now() - timedelta(hours=2)
                
        except Exception as e:
            logging.warning(f"Failed to get watermark for {tech_center}, using default: {e}")
            return datetime.now() - timedelta(hours=2)
    
    def _predict_incidents(self, incidents: pd.DataFrame, models: Dict[str, Any], tech_center: str) -> List[Dict[str, Any]]:
        """
        Predict cluster assignments for new incidents
        """
        self.logger.log_stage_start("incident_prediction", {
            "incident_count": len(incidents),
            "tech_center": tech_center
        })
        
        predictions = []
        
        for idx, incident in incidents.iterrows():
            try:
                # Process text and generate embeddings
                processed_text = self.text_processor.process_incident_text(
                    incident["description"], 
                    incident["short_description"]
                )
                
                # Generate embedding
                embedding = self.embedding_generator.generate_single_embedding(processed_text)
                
                # Apply UMAP transformation
                umap_embedding = models["umap"].transform([embedding])
                
                # Predict cluster using HDBSCAN
                cluster_label = models["hdbscan"].predict(umap_embedding)[0]
                
                # Get cluster metadata
                cluster_info = self._get_cluster_info(cluster_label, models["metadata"])
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(
                    umap_embedding[0], cluster_label, models["hdbscan"]
                )
                
                prediction = {
                    "incident_id": incident["incident_id"],
                    "tech_center": tech_center,
                    "cluster_label": int(cluster_label),
                    "cluster_name": cluster_info.get("cluster_name", f"Cluster_{cluster_label}"),
                    "domain": cluster_info.get("domain", "Unknown"),
                    "confidence_score": float(confidence_score),
                    "is_outlier": cluster_label == -1,
                    "prediction_timestamp": datetime.now(),
                    "model_version": f"{models['metadata'].get('year', 'unknown')}_{models['metadata'].get('quarter', 'unknown')}",
                    "embedding_vector": embedding.tolist(),
                    "umap_coordinates": umap_embedding[0].tolist()
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logging.error(f"Failed to predict incident {incident['incident_id']}: {e}")
                # Add failed prediction record
                predictions.append({
                    "incident_id": incident["incident_id"],
                    "tech_center": tech_center,
                    "cluster_label": -1,
                    "cluster_name": "Prediction_Failed",
                    "domain": "Error",
                    "confidence_score": 0.0,
                    "is_outlier": True,
                    "prediction_timestamp": datetime.now(),
                    "model_version": "error",
                    "error_message": str(e)
                })
        
        self.logger.log_stage_complete("incident_prediction", {
            "successful_predictions": len([p for p in predictions if "error_message" not in p]),
            "failed_predictions": len([p for p in predictions if "error_message" in p])
        })
        
        return predictions
    
    def _get_cluster_info(self, cluster_label: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get cluster information from metadata
        """
        clusters = metadata.get("clusters", {})
        return clusters.get(str(cluster_label), {})
    
    def _calculate_confidence_score(self, umap_point: np.ndarray, cluster_label: int, hdbscan_model) -> float:
        """
        Calculate confidence score for prediction
        """
        try:
            if cluster_label == -1:  # Outlier
                return 0.0
            
            # Use HDBSCAN outlier scores or membership probabilities if available
            if hasattr(hdbscan_model, 'outlier_scores_'):
                # Higher outlier score means less confidence
                outlier_score = hdbscan_model.outlier_scores_[0] if len(hdbscan_model.outlier_scores_) > 0 else 0.5
                return max(0.0, 1.0 - outlier_score)
            
            # Fallback: simple distance-based confidence
            return 0.7  # Default confidence
            
        except Exception:
            return 0.5  # Default confidence if calculation fails
    
    def _store_predictions(self, predictions: List[Dict[str, Any]], tech_center: str):
        """
        Store predictions to BigQuery
        """
        if not predictions:
            return
        
        try:
            # Convert to DataFrame
            df_predictions = pd.DataFrame(predictions)
            
            # Store to BigQuery
            table_id = f"{self.config.bigquery.project_id}.{self.config.bigquery.datasets.results}.incident_predictions"
            
            self.bigquery_client.upload_dataframe(
                df_predictions, 
                table_id, 
                write_disposition="WRITE_APPEND"
            )
            
            logging.info(f"Stored {len(predictions)} predictions for {tech_center}")
            
            # Also save to local if configured
            if self.config.pipeline.save_to_local:
                self._save_predictions_locally(predictions, tech_center)
                
        except Exception as e:
            raise RuntimeError(f"Failed to store predictions for {tech_center}: {e}")
    
    def _save_predictions_locally(self, predictions: List[Dict[str, Any]], tech_center: str):
        """
        Save predictions to local file system
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{tech_center}_{timestamp}.json"
            
            local_path = os.path.join(
                self.config.pipeline.result_path,
                "predictions",
                tech_center,
                filename
            )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Save predictions
            import json
            with open(local_path, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            
            logging.info(f"Saved predictions locally: {local_path}")
            
        except Exception as e:
            logging.error(f"Failed to save predictions locally for {tech_center}: {e}")
    
    def _update_prediction_watermark(self, tech_center: str, incidents: pd.DataFrame):
        """
        Update the prediction watermark timestamp
        """
        try:
            if len(incidents) == 0:
                return
            
            # Get the latest incident timestamp
            latest_timestamp = incidents["sys_created_on"].max()
            
            # Update watermark in BigQuery
            watermark_data = {
                "tech_center": tech_center,
                "prediction_timestamp": datetime.now(),
                "last_processed_incident_time": latest_timestamp,
                "processed_incident_count": len(incidents)
            }
            
            df_watermark = pd.DataFrame([watermark_data])
            table_id = f"{self.config.bigquery.project_id}.{self.config.bigquery.datasets.results}.prediction_watermarks"
            
            self.bigquery_client.upload_dataframe(
                df_watermark,
                table_id,
                write_disposition="WRITE_APPEND"
            )
            
            logging.info(f"Updated prediction watermark for {tech_center}")
            
        except Exception as e:
            logging.error(f"Failed to update watermark for {tech_center}: {e}")
    
    def _get_cluster_distribution(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get distribution of predictions across clusters
        """
        distribution = {}
        for prediction in predictions:
            cluster_name = prediction.get("cluster_name", "Unknown")
            distribution[cluster_name] = distribution.get(cluster_name, 0) + 1
        
        return distribution
    
    @catch_errors
    def get_prediction_status(self, tech_center: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of prediction pipeline
        """
        try:
            if tech_center:
                # Status for specific tech center
                watermark = self._get_prediction_watermark(tech_center)
                model_info = self._get_latest_model_info(tech_center)
                
                return {
                    "tech_center": tech_center,
                    "last_prediction": watermark.isoformat(),
                    "model_info": model_info,
                    "status": "active" if model_info else "no_model"
                }
            else:
                # Status for all tech centers
                status = {
                    "timestamp": datetime.now().isoformat(),
                    "tech_centers": {},
                    "summary": {
                        "total_centers": len(self.config.pipeline.tech_centers),
                        "active_centers": 0,
                        "inactive_centers": 0
                    }
                }
                
                for tc in self.config.pipeline.tech_centers:
                    tc_status = self.get_prediction_status(tc)
                    status["tech_centers"][tc] = tc_status
                    
                    if tc_status["status"] == "active":
                        status["summary"]["active_centers"] += 1
                    else:
                        status["summary"]["inactive_centers"] += 1
                
                return status
                
        except Exception as e:
            raise RuntimeError(f"Failed to get prediction status: {e}")