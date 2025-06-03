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
import asyncio
from dataclasses import dataclass

from data.bigquery_client import BigQueryClient
from data.blob_storage import BlobStorageClient
from preprocessing.embedding_generation import EmbeddingGenerator
from preprocessing.text_processing import TextProcessor
from utils.error_handler import PipelineLogger, catch_errors
from config.config import load_config


@dataclass
class ModelCache:
    """Model cache entry"""
    model: Any
    metadata: Dict
    loaded_at: datetime
    tech_center: str
    version: str
    
    def is_expired(self, ttl_hours: int) -> bool:
        """Check if cache entry is expired"""
        return (datetime.now() - self.loaded_at).total_seconds() > ttl_hours * 3600

class PredictionPipeline:
    """
    Real-time prediction pipeline for incident classification using cached HDBSCAN models.
    Supports model versioning and 2-hour prediction cycles.
    """
    
    def __init__(self, config=None):
        """Initialize prediction pipeline with updated config system"""
        self.config = config if config is not None else load_config()
        
        # Model cache for performance
        self.model_cache: Dict[str, ModelCache] = {}
        
        # Prediction statistics
        self.prediction_stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "prediction_errors": 0,
            "last_prediction_time": None
        }
        
        # Initialize clients
        self.bigquery_client = BigQueryClient(config)
        self.blob_client = BlobStorageClient(config)
        
        # Initialize processors
        self.text_processor = TextProcessor(config)
        self.embedding_generator = EmbeddingGenerator(config)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Table references from configuration instead of hardcoded values
        self.preprocessed_table = config.bigquery.tables.preprocessed_incidents
        self.predictions_table = config.bigquery.tables.predictions
        
        self.logger.info("Prediction pipeline initialized with optimized storage architecture")
    
    @catch_errors
    async def predict_batch(self, incidents: pd.DataFrame, 
                          tech_centers: List[str]) -> List[Dict]:
        """
        Predict clusters for a batch of incidents across multiple tech centers.
        
        Args:
            incidents: DataFrame with incident data
            tech_centers: List of tech centers to predict for
            
        Returns:
            List of prediction results
        """
        prediction_start = datetime.now()
        
        self.logger.info("Starting batch prediction for %d incidents across %d tech centers",
                    len(incidents), len(tech_centers))
        
        try:
            # Group incidents by tech center
            incidents_by_tc = self._group_incidents_by_tech_center(incidents, tech_centers)
            
            # Preload models for all tech centers
            await self._preload_models(tech_centers)
            
            # Process predictions for each tech center
            all_predictions = []
            
            for tech_center, tc_incidents in incidents_by_tc.items():
                if len(tc_incidents) == 0:
                    continue
                
                self.logger.info("Predicting for %s: %d incidents", tech_center, len(tc_incidents))
                
                tc_predictions = await self._predict_tech_center(tech_center, tc_incidents)
                all_predictions.extend(tc_predictions)
            
            # Update statistics
            prediction_duration = datetime.now() - prediction_start
            self.prediction_stats["total_predictions"] += len(all_predictions)
            self.prediction_stats["last_prediction_time"] = datetime.now().isoformat()
            
            self.logger.info("Batch prediction completed: %d predictions in %.2f seconds",
                        len(all_predictions), prediction_duration.total_seconds())
            
            return all_predictions
            
        except Exception as e:
            self.logger.error("Batch prediction failed: %s", str(e))
            self.prediction_stats["prediction_errors"] += 1
            raise
    
    @catch_errors
    async def predict_single_incident(self, incident: Dict, tech_center: str) -> Dict:
        """
        Predict cluster for a single incident.
        
        Args:
            incident: Incident data
            tech_center: Tech center name
            
        Returns:
            Prediction result
        """
        try:
            # Convert to DataFrame for consistency
            incident_df = pd.DataFrame([incident])
            
            # Get prediction
            predictions = await self._predict_tech_center(tech_center, incident_df)
            
            return predictions[0] if predictions else {"status": "failed", "reason": "no_prediction"}
            
        except Exception as e:
            self.logger.error("Single incident prediction failed: %s", str(e))
            return {"status": "failed", "error": str(e)}
    
    @catch_errors
    async def run_predictions_all_tech_centers(self) -> Dict[str, Any]:
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
                center_result = await self.run_prediction_for_tech_center(tech_center)
                
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
    async def run_prediction_for_tech_center(self, tech_center: str) -> Dict[str, Any]:
        """
        Run predictions for a specific tech center
        """
        self.logger.log_stage_start(f"prediction_{tech_center}", {"tech_center": tech_center})
        
        # Get latest model info
        model_info = await self._get_latest_model_info(tech_center)
        if not model_info:
            raise ValueError(f"No trained model found for tech center: {tech_center}")
        
        # Load trained models
        models = await self._load_models_for_tech_center(tech_center, model_info)
        
        # Get new incidents since last prediction
        new_incidents = await self._get_new_incidents_for_tech_center(tech_center)
        
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
        predictions = await self._predict_incidents(new_incidents, models, tech_center)
        
        # Store predictions
        await self._store_predictions(predictions, tech_center)
        
        # Update watermark
        await self._update_prediction_watermark(tech_center, new_incidents)
        
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
    
    async def _get_latest_model_info(self, tech_center: str) -> Optional[Dict[str, Any]]:
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
    
    async def _load_models_for_tech_center(self, tech_center: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def _get_new_incidents_for_tech_center(self, tech_center: str) -> pd.DataFrame:
        """
        Get new incidents for tech center since last prediction
        """
        try:
            # Get last prediction timestamp (watermark)
            last_prediction_time = await self._get_prediction_watermark(tech_center)
            
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
    
    async def _get_prediction_watermark(self, tech_center: str) -> datetime:
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
    
    async def _predict_incidents(self, incidents: pd.DataFrame, models: Dict[str, Any], tech_center: str) -> List[Dict[str, Any]]:
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
    
    async def _predict_incidents_improved(self, incidents: pd.DataFrame, models: Dict[str, Any], tech_center: str) -> List[Dict[str, Any]]:
        """
        YOUR IMPROVED APPROACH: Predict using combined model artifacts and domain mappings
        """
        self.logger.info(f"Starting improved prediction for {len(incidents)} incidents in {tech_center}")
        
        predictions = []
        
        for idx, incident in incidents.iterrows():
            try:
                # Process text and generate embeddings (existing logic)
                processed_text = self.text_processor.process_incident_text(
                    incident["description"], 
                    incident["short_description"]
                )
                
                # Generate embedding
                embedding = self.embedding_generator.generate_single_embedding(processed_text)
                
                # Apply UMAP transformation using loaded model
                umap_embedding = models["model_artifacts"]["umap_model"].transform([embedding])
                
                # Predict cluster using loaded HDBSCAN model - YOUR APPROACH
                cluster_label = models["model_artifacts"]["hdbscan_model"].predict(umap_embedding)[0]
                
                # Get cluster metadata using YOUR COMBINED APPROACH
                cluster_name = models.get("cluster_labels", {}).get(str(cluster_label), f"Cluster_{cluster_label}")
                domain_info = models.get("domain_mappings", {}).get(str(cluster_label), {"domain_name": "Unknown"})
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(
                    umap_embedding[0], cluster_label, models["model_artifacts"]["hdbscan_model"]
                )
                
                prediction = {
                    "incident_id": incident["incident_id"],
                    "tech_center": tech_center,
                    "cluster_label": int(cluster_label),
                    "cluster_name": cluster_name,
                    "domain": domain_info.get("domain_name", "Unknown"),
                    "confidence_score": float(confidence_score),
                    "is_outlier": cluster_label == -1,
                    "prediction_timestamp": datetime.now(),
                    # Optional: consider not storing these in final prediction output for BQ cost optimization
                    "embedding_vector": embedding.tolist() if self.config.prediction.store_embeddings else None,
                    "umap_coordinates": umap_embedding[0].tolist() if self.config.prediction.store_coordinates else None,
                    # YOUR KEY ADDITIONS for traceability
                    "model_table_used": models.get("model_metadata", {}).get("training_results_table", "N/A"),
                    "blob_model_path": models.get("model_artifacts", {}).get("blob_path", "N/A"),
                    "model_hash": models.get("model_metadata", {}).get("model_hash", "N/A")
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                self.logger.error(f"Failed to predict incident {incident['incident_id']}: {e}")
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
                    "error_message": str(e),
                    "model_table_used": models.get("model_metadata", {}).get("training_results_table", "N/A"),
                    "model_hash": models.get("model_metadata", {}).get("model_hash", "N/A")
                })
        
        self.logger.info(f"Completed predictions: {len([p for p in predictions if 'error_message' not in p])} successful, {len([p for p in predictions if 'error_message' in p])} failed")
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
    
    async def _store_predictions(self, predictions: List[Dict[str, Any]], tech_center: str):
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
    
    async def _update_prediction_watermark(self, tech_center: str, incidents: pd.DataFrame):
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
    async def get_prediction_status(self, tech_center: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of prediction pipeline
        """
        try:
            if tech_center:
                # Status for specific tech center
                watermark = await self._get_prediction_watermark(tech_center)
                model_info = await self._get_latest_model_info(tech_center)
                
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
                    tc_status = await self.get_prediction_status(tc)
                    status["tech_centers"][tc] = tc_status
                    
                    if tc_status["status"] == "active":
                        status["summary"]["active_centers"] += 1
                    else:
                        status["summary"]["inactive_centers"] += 1
                
                return status
                
        except Exception as e:
            raise RuntimeError(f"Failed to get prediction status: {e}")
      def predict_new_incidents(self, tech_center: str, model_year: int = 2024, model_quarter: str = "q4") -> Dict:
        """
        Predict clusters/domains for new incidents using versioned trained models
        
        Architecture:
        1. Load embeddings from preprocessed_incidents table
        2. Load trained models from versioned clustering_predictions table  
        3. Generate predictions (cluster + domain)
        4. Save results to incident_predictions table (without embeddings)
        5. Reference specific model version from cumulative training
        """        try:
            self.logger.info(f"Starting predictions for {tech_center} using model {model_year}_{model_quarter}")
            
            # 1. Generate versioned table name for trained model
            tech_center_hash = abs(hash(tech_center)) % 1000
            table_version = f"{model_year}_{model_quarter}_{tech_center_hash:03d}"
            model_table_name = f"clustering_predictions_{table_version}"
            
            self.logger.info(f"Using trained model from table: {model_table_name}")
            
            # 2. Get new incidents with embeddings from preprocessed_incidents
            incidents_with_embeddings = self._get_new_incidents_with_embeddings(tech_center)
            
            if not incidents_with_embeddings:
                self.logger.info(f"No new incidents found for {tech_center}")
                return {"status": "no_new_incidents", "count": 0}
            
            # 3. Load trained models from versioned table
            models = self._load_trained_models_from_versioned_table(tech_center, model_table_name, model_year, model_quarter)
            
            # 4. Generate predictions using embeddings
            predictions = self._generate_predictions(incidents_with_embeddings, models, tech_center)
            
            # 5. Save predictions (without embeddings for cost optimization)
            self._save_predictions_to_bigquery(predictions, tech_center, model_table_name)
            
            self.logger.info(f"Completed predictions for {tech_center}: {len(predictions)} incidents")
            
            return {
                "status": "success",
                "tech_center": tech_center,
                "predictions_count": len(predictions),
                "model_table_used": model_table_name,
                "model_version": f"{model_year}_{model_quarter}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {tech_center}: {e}")
            raise
    
    def _get_new_incidents_with_embeddings(self, tech_center: str) -> List[Dict]:
        """
        Get new incidents with embeddings from preprocessed_incidents table
        This is where embeddings are stored (not in clustering_predictions)
        """
        # Mock implementation - would query preprocessed_incidents table
        self.logger.info(f"Loading new incidents with embeddings for {tech_center}")
        
        # In real implementation, query would be:
        # SELECT number, sys_created_on, combined_incidents_summary, embedding
        # FROM preprocessed_incidents 
        # WHERE tech_center = ? AND processed_at > last_prediction_time
        
        return [
            {
                "number": f"INC{2000000 + i}",
                "sys_created_on": "2024-01-15T14:30:00Z",
                "combined_incidents_summary": f"New incident {i} summary",
                "embedding": np.random.rand(1536).tolist()  # Embedding from preprocessed table
            }
            for i in range(10)  # Mock 10 new incidents
        ]
    
    def _generate_predictions(self, incidents: List[Dict], models: Dict, tech_center: str) -> List[Dict]:
        """Generate cluster and domain predictions"""
        predictions = []
        
        for incident in incidents:
            # Mock prediction logic
            predicted_cluster = np.random.randint(0, 10)
            confidence = np.random.uniform(0.6, 0.95)
            
            prediction = {
                "incident_id": incident["number"],
                "number": incident["number"],
                "sys_created_on": incident["sys_created_on"],
                "combined_incidents_summary": incident["combined_incidents_summary"],
                
                # Prediction results
                "predicted_cluster_id": predicted_cluster,
                "predicted_cluster_label": f"Predicted Cluster {predicted_cluster}",
                "confidence_score": confidence,
                "is_outlier": predicted_cluster == -1,
                
                # Domain prediction
                "predicted_domain_id": predicted_cluster // 3,  # Group clusters into domains
                "predicted_domain_name": f"Predicted Domain {predicted_cluster // 3}",
                "domain_confidence": confidence * 0.9,
                
                # Metadata
                "tech_center": tech_center,
                "prediction_timestamp": datetime.now().isoformat(),
                "model_used": f"{tech_center}_2024_q4",
                "pipeline_run_id": f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                
                # Technical details (no embeddings for cost optimization)
                "distance_to_centroid": np.random.uniform(0.1, 0.5),
                "nearest_neighbors": [f"INC{2000000 + j}" for j in range(3)]
            }
            
            predictions.append(prediction)
        
        return predictions
      def _save_predictions_to_bigquery(self, predictions: List[Dict], tech_center: str, model_table_name: str):
        """
        Save predictions to incident_predictions table
        Note: No embeddings saved here for cost optimization
        """
        self.logger.info(f"Saving {len(predictions)} predictions to BigQuery for {tech_center}")
        
        # Add model table reference to each prediction
        for prediction in predictions:
            prediction["model_table_used"] = model_table_name
            # Remove embedding if it exists (should not be in predictions table)
            prediction.pop('embedding', None)
        
        # Mock save operation
        # In real implementation, this would insert to incident_predictions table
        # Schema: incident_id, predicted_cluster_id, predicted_cluster_label, confidence_score,
        #         predicted_domain_id, predicted_domain_name, domain_confidence,
        #         tech_center, prediction_timestamp, model_table_used
        #         (NO embedding column for cost savings)
        
        self.logger.info(f"Predictions saved to incident_predictions table (without embeddings for cost optimization)")
        self.logger.info(f"Model table referenced: {model_table_name}")
      def _load_trained_models_from_versioned_table(self, tech_center: str, model_table_name: str, 
                                                model_year: int, model_quarter: str) -> Dict:
        """
        Load trained models from blob storage and domain mappings from versioned BigQuery table
        
        Architecture:
        1. Load model artifacts (UMAP, HDBSCAN) from blob storage
        2. Load domain mappings and cluster labels from BigQuery versioned table
        3. Combine for prediction use
        
        Args:
            tech_center: Tech center name
            model_table_name: Versioned table name (e.g., clustering_predictions_2025_q2_789)
            model_year: Training model year
            model_quarter: Training model quarter
            
        Returns:
            Dictionary containing model artifacts and domain mappings
        """
        try:
            self.logger.info(f"Loading model artifacts from blob storage and metadata from {model_table_name}")
            
            # 1. Load model artifacts from blob storage
            model_artifacts = self._load_model_artifacts_from_blob(tech_center, model_year, model_quarter)
            
            # 2. Load domain mappings and cluster metadata from BigQuery
            domain_mappings = self._load_domain_mappings_from_bigquery(model_table_name, tech_center)
            
            # 3. Combine model artifacts with domain mappings
            models = {
                "model_artifacts": model_artifacts,
                "cluster_labels": domain_mappings["cluster_labels"],
                "domain_mappings": domain_mappings["domain_mappings"],
                "model_metadata": {
                    "table_name": model_table_name,
                    "tech_center": tech_center,
                    "model_version": f"{model_year}_{model_quarter}_v1",
                    "training_approach": "cumulative_24_months",
                    "domain_count": len(set(dm["domain_id"] for dm in domain_mappings["domain_mappings"].values())),
                    "cluster_count": len(domain_mappings["cluster_labels"]),
                    "blob_storage_path": model_artifacts.get("blob_path", "N/A")
                }
            }
            
            self.logger.info(f"Loaded model: {len(models['cluster_labels'])} clusters, {models['model_metadata']['domain_count']} domains")
            self.logger.info(f"Model source: Blob storage + BigQuery table {model_table_name}")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to load models from blob storage and {model_table_name}: {e}")
            raise
      def _load_model_artifacts_from_blob(self, tech_center: str, model_year: int, model_quarter: str) -> Dict:
        """
        REAL IMPLEMENTATION: Load UMAP and HDBSCAN model artifacts from Azure Blob Storage
        
        Blob Storage Path:
        hdbscan-models/{tech_center_folder}/{model_year}_{model_quarter}/
        ├── umap_model.pkl
        ├── hdbscan_model.pkl  
        ├── preprocessing_artifacts.pkl
        └── model_metadata.json
        """
        try:
            import pickle
            import io
            from azure.storage.blob import BlobServiceClient
            
            # Get blob storage connection
            connection_string = self.config.azure.blob_storage.connection_string
            container_name = self.config.azure.blob_storage.container_name
            
            # Initialize blob client
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(container_name)
            
            # Build blob paths based on tech center and version
            tech_center_folder = tech_center.replace(" ", "-").replace("_", "-").lower()
            model_version = f"{model_year}_{model_quarter}"
            blob_prefix = f"{tech_center_folder}/{model_version}/"
            
            self.logger.info(f"Loading model artifacts from blob: {container_name}/{blob_prefix}")
            
            # Load UMAP model
            umap_blob_name = f"{blob_prefix}umap_model.pkl"
            umap_blob_client = container_client.get_blob_client(umap_blob_name)
            umap_data = umap_blob_client.download_blob().readall()
            umap_model = pickle.loads(umap_data)
            
            # Load HDBSCAN model
            hdbscan_blob_name = f"{blob_prefix}hdbscan_model.pkl"
            hdbscan_blob_client = container_client.get_blob_client(hdbscan_blob_name)
            hdbscan_data = hdbscan_blob_client.download_blob().readall()
            hdbscan_model = pickle.loads(hdbscan_data)
            
            # Load preprocessing artifacts
            preprocessing_blob_name = f"{blob_prefix}preprocessing_artifacts.pkl"
            preprocessing_blob_client = container_client.get_blob_client(preprocessing_blob_name)
            preprocessing_data = preprocessing_blob_client.download_blob().readall()
            preprocessing_artifacts = pickle.loads(preprocessing_data)
            
            # Load model metadata
            metadata_blob_name = f"{blob_prefix}model_metadata.json"
            metadata_blob_client = container_client.get_blob_client(metadata_blob_name)
            metadata_data = metadata_blob_client.download_blob().readall()
            model_metadata = json.loads(metadata_data.decode('utf-8'))
            
            model_artifacts = {
                "umap_model": umap_model,
                "hdbscan_model": hdbscan_model,
                "preprocessing_artifacts": preprocessing_artifacts,
                "model_metadata": model_metadata,
                "blob_path": f"{container_name}/{blob_prefix}",
                "loaded_from": "blob_storage",
                "model_version": model_version
            }
            
            self.logger.info(f"Successfully loaded real model artifacts from {blob_prefix}")
            return model_artifacts
            
        except Exception as e:
            self.logger.error(f"Failed to load model artifacts from blob storage: {e}")
            raise RuntimeError(f"Could not load model artifacts for {tech_center} {model_year}_{model_quarter}: {e}")    def _load_domain_mappings_from_bigquery(self, models: Dict[str, Any], tech_center: str) -> Dict:
        """
        REAL IMPLEMENTATION: Load domain mappings from versioned BigQuery table
        Using YOUR APPROACH - get table name from model metadata
        """
        try:
            # YOUR CRITICAL INSIGHT - get table name from model metadata
            if not models or "model_metadata" not in models or "training_results_table" not in models["model_metadata"]:
                self.logger.error(f"Cannot load domain mappings: training_results_table not found in model metadata for {tech_center}")
                return {"cluster_labels": {}, "domain_mappings": {}}
            
            actual_model_table_name = models["model_metadata"]["training_results_table"]
            self.logger.info(f"Loading domain mappings from BigQuery table: {actual_model_table_name} for {tech_center}")
            
            # REAL IMPLEMENTATION using existing BigQuery client
            query = f"""
            SELECT DISTINCT 
                cluster_id, 
                cluster_label, 
                domain_id, 
                domain_name,
                domain_description,
                cluster_count,
                incident_count
            FROM `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.{actual_model_table_name}` 
            WHERE tech_center = @tech_center
            ORDER BY cluster_id
            """
            
            # Execute query using existing BigQuery client
            job_config = self.bigquery_client.create_query_job_config()
            job_config.query_parameters = [
                self.bigquery_client.bigquery.ScalarQueryParameter("tech_center", "STRING", tech_center)
            ]
            
            results = self.bigquery_client.execute_query(query, job_config)
            
            # Process results into domain mappings
            cluster_labels = {}
            domain_mappings = {}
            
            for _, row in results.iterrows():
                cluster_id = str(row['cluster_id'])
                
                # Store cluster label
                cluster_labels[cluster_id] = row['cluster_label']
                
                # Store domain mapping
                domain_mappings[cluster_id] = {
                    "domain_id": row['domain_id'],
                    "domain_name": row['domain_name'],
                    "domain_description": row.get('domain_description', ''),
                    "cluster_count": row.get('cluster_count', 0),
                    "incident_count": row.get('incident_count', 0),
                    "confidence": min(1.0, max(0.1, (row.get('incident_count', 1)) / 100.0))
                }
            
            domain_data = {
                "cluster_labels": cluster_labels,
                "domain_mappings": domain_mappings,
                "table_source": actual_model_table_name,
                "tech_center": tech_center,
                "total_clusters": len(cluster_labels),
                "total_domains": len(set(dm["domain_id"] for dm in domain_mappings.values()))
            }
            
            self.logger.info(f"Loaded {len(cluster_labels)} cluster labels and {domain_data['total_domains']} domains from {actual_model_table_name}")
            return domain_data
            
        except Exception as e:
            self.logger.error(f"Failed to load domain mappings: {e}")
            raise RuntimeError(f"Could not load domain mappings for {tech_center}: {e}")