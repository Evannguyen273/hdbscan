import logging
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib
import json

from config.config import get_config, get_current_quarter
from .training_pipeline import TrainingPipeline
from .prediction_pipeline import PredictionPipeline
from .preprocessing_pipeline import PreprocessingPipeline
from data_access.bigquery_client import BigQueryClient
from storage.azure_blob_client import AzureBlobClient

class PipelineOrchestrator:
    """
    Main orchestrator for cumulative HDBSCAN pipeline with versioned storage.
    Handles semi-annual training cycles and real-time predictions.
    """
    
    def __init__(self, config=None):
        """Initialize pipeline orchestrator with new config system"""
        self.config = config if config is not None else get_config()
        
        # Initialize pipeline components
        self.training_pipeline = TrainingPipeline(self.config)
        self.prediction_pipeline = PredictionPipeline(self.config)
        self.preprocessing_pipeline = PreprocessingPipeline(self.config)
        
        # Initialize storage clients
        self.bigquery_client = BigQueryClient(self.config)
        self.blob_client = AzureBlobClient(self.config)
        
        # Pipeline state
        self.current_models = {}
        self.pipeline_stats = {}
        
        logging.info("Pipeline orchestrator initialized with cumulative training approach")
    
    async def run_training_cycle(self, year: int, quarter: str, force_retrain: bool = False) -> Dict:
        """
        Run a complete training cycle for cumulative HDBSCAN approach.
        
        Args:
            year: Training year (e.g., 2025)
            quarter: Training quarter ('q1', 'q2', 'q3', 'q4')  
            force_retrain: Force retraining even if model exists
            
        Returns:
            Training results with model metadata
        """
        training_start = datetime.now()
        version = f"{year}_{quarter}"
        
        logging.info("=" * 80)
        logging.info("STARTING CUMULATIVE TRAINING CYCLE")
        logging.info("=" * 80)
        logging.info("Version: %s", version)
        logging.info("Training window: %d months", self.config.training.training_window_months)
        
        try:
            # Check if model already exists
            if not force_retrain and await self._model_exists(version):
                logging.info("Model version %s already exists, skipping training", version)
                return {"status": "skipped", "version": version, "reason": "model_exists"}
            
            # Stage 1: Prepare cumulative dataset
            logging.info("Stage 1: Preparing cumulative dataset...")
            dataset = await self._prepare_cumulative_dataset(year, quarter)
            
            if len(dataset) == 0:
                logging.error("No data available for training")
                return {"status": "failed", "version": version, "reason": "no_data"}
            
            # Stage 2: Run preprocessing pipeline
            logging.info("Stage 2: Running preprocessing pipeline...")
            preprocessing_results = await self.preprocessing_pipeline.process_for_training(dataset)
            
            # Stage 3: Run training for all tech centers
            logging.info("Stage 3: Running training for all tech centers...")
            training_results = await self._run_parallel_training(
                preprocessing_results, version, year, quarter
            )
            
            # Stage 4: Store models and update registry
            logging.info("Stage 4: Storing models and updating registry...")
            storage_results = await self._store_models_and_update_registry(
                training_results, version, year, quarter
            )
            
            training_duration = datetime.now() - training_start
            
            final_results = {
                "status": "success",
                "version": version,
                "training_duration_seconds": training_duration.total_seconds(),
                "models_trained": len(training_results),
                "tech_centers": list(training_results.keys()),
                "storage_results": storage_results,
                "training_window_months": self.config.training.training_window_months
            }
            
            logging.info("Training cycle completed successfully in %.2f minutes", 
                        training_duration.total_seconds() / 60)
            
            return final_results
            
        except Exception as e:
            logging.error("Training cycle failed: %s", str(e))
            return {
                "status": "failed", 
                "version": version,
                "error": str(e),
                "training_duration_seconds": (datetime.now() - training_start).total_seconds()
            }
    
    async def run_prediction_cycle(self, tech_centers: Optional[List[str]] = None) -> Dict:
        """
        Run prediction cycle for real-time incident classification.
        
        Args:
            tech_centers: Specific tech centers to predict for (default: all)
            
        Returns:
            Prediction results
        """
        prediction_start = datetime.now()
        
        logging.info("=" * 80)
        logging.info("STARTING PREDICTION CYCLE")
        logging.info("=" * 80)
        
        try:
            # Get tech centers to process
            if tech_centers is None:
                tech_centers = self.config.tech_centers
            
            # Get unprocessed incidents
            incidents = await self._get_unprocessed_incidents(tech_centers)
            
            if len(incidents) == 0:
                logging.info("No unprocessed incidents found")
                return {"status": "success", "incidents_processed": 0}
            
            logging.info("Processing %d unprocessed incidents", len(incidents))
            
            # Run prediction pipeline
            prediction_results = await self.prediction_pipeline.predict_batch(
                incidents, tech_centers
            )
            
            # Store predictions
            storage_results = await self._store_predictions(prediction_results)
            
            prediction_duration = datetime.now() - prediction_start
            
            final_results = {
                "status": "success",
                "prediction_duration_seconds": prediction_duration.total_seconds(),
                "incidents_processed": len(incidents),
                "tech_centers_processed": len(tech_centers),
                "predictions_stored": storage_results.get("rows_inserted", 0)
            }
            
            logging.info("Prediction cycle completed in %.2f seconds", 
                        prediction_duration.total_seconds())
            
            return final_results
            
        except Exception as e:
            logging.error("Prediction cycle failed: %s", str(e))
            return {
                "status": "failed",
                "error": str(e),
                "prediction_duration_seconds": (datetime.now() - prediction_start).total_seconds()
            }
    
    async def run_scheduled_training(self) -> Dict:
        """
        Run training based on configured schedule (semi-annual).
        
        Returns:
            Training results if training is due, else skip status
        """
        current_month = datetime.now().month
        current_year = datetime.now().year
        current_quarter = get_current_quarter()
        
        # Check if training is due based on schedule
        training_months = self.config.training.schedule['months']
        
        if current_month in training_months:
            logging.info("Training is due for month %d (quarter %s)", current_month, current_quarter)
            return await self.run_training_cycle(current_year, current_quarter)
        else:
            next_training_month = min([m for m in training_months if m > current_month] + 
                                     [m + 12 for m in training_months])
            logging.info("Training not due. Next training in month %d", next_training_month % 12)
            return {
                "status": "skipped", 
                "reason": "not_scheduled",
                "next_training_month": next_training_month % 12
            }
    
    async def _prepare_cumulative_dataset(self, year: int, quarter: str) -> pd.DataFrame:
        """Prepare cumulative dataset with specified training window"""
        window_months = self.config.training.training_window_months
        
        # Calculate date range for cumulative approach
        end_date = self._get_quarter_end_date(year, quarter)
        start_date = end_date - timedelta(days=window_months * 30)  # Approximate months
        
        logging.info("Fetching cumulative data from %s to %s (%d months)", 
                    start_date.date(), end_date.date(), window_months)
        
        # Fetch data from BigQuery
        query = f"""
        SELECT *
        FROM `{self.config.bigquery.tables['raw_incidents']}`
        WHERE created_date >= '{start_date.date()}'
        AND created_date <= '{end_date.date()}'
        AND tech_center IN ({','.join([f"'{tc}'" for tc in self.config.tech_centers])})
        ORDER BY created_date
        """
        
        return await self.bigquery_client.query_to_dataframe(query)
    
    async def _run_parallel_training(self, preprocessing_results: Dict, 
                                   version: str, year: int, quarter: str) -> Dict:
        """Run training for all tech centers in parallel"""
        max_workers = self.config.training.processing['max_workers']
        
        training_tasks = []
        for tech_center in self.config.tech_centers:
            if tech_center in preprocessing_results:
                task = self.training_pipeline.train_tech_center(
                    tech_center, preprocessing_results[tech_center], version, year, quarter
                )
                training_tasks.append((tech_center, task))
        
        # Run training tasks with concurrency limit
        results = {}
        semaphore = asyncio.Semaphore(max_workers)
        
        async def bounded_training(tech_center: str, task):
            async with semaphore:
                return await task
        
        for tech_center, task in training_tasks:
            try:
                result = await bounded_training(tech_center, task)
                results[tech_center] = result
                logging.info("Training completed for %s", tech_center)
            except Exception as e:
                logging.error("Training failed for %s: %s", tech_center, str(e))
                results[tech_center] = {"status": "failed", "error": str(e)}
        
        return results
    
    async def _store_models_and_update_registry(self, training_results: Dict, 
                                              version: str, year: int, quarter: str) -> Dict:
        """Store trained models and update model registry"""
        storage_results = {}
        
        for tech_center, result in training_results.items():
            if result.get("status") == "success":
                # Generate model hash
                model_hash = self._generate_model_hash(result["model_data"])
                
                # Store model in Azure Blob
                blob_path = f"models/{version}/{tech_center}/{model_hash}.pkl"
                await self.blob_client.upload_model(result["model_data"], blob_path)
                
                # Update BigQuery model registry
                registry_entry = {
                    "version": version,
                    "tech_center": tech_center,
                    "model_hash": model_hash,
                    "blob_path": blob_path,
                    "training_date": datetime.now().isoformat(),
                    "year": year,
                    "quarter": quarter,
                    "training_window_months": self.config.training.training_window_months,
                    "model_metrics": result.get("metrics", {}),
                    "status": "active"
                }
                
                await self.bigquery_client.insert_model_registry_entry(registry_entry)
                storage_results[tech_center] = {"status": "success", "hash": model_hash}
            else:
                storage_results[tech_center] = {"status": "failed"}
        
        return storage_results
    
    def _generate_model_hash(self, model_data: Dict) -> str:
        """Generate hash for model versioning"""
        hash_algorithm = self.config.training.versioning['hash_algorithm']
        hash_length = self.config.training.versioning['hash_length']
        
        # Create deterministic hash from model data
        model_str = json.dumps(model_data, sort_keys=True, default=str)
        hash_obj = hashlib.new(hash_algorithm)
        hash_obj.update(model_str.encode('utf-8'))
        
        return hash_obj.hexdigest()[:hash_length]
    
    def _get_quarter_end_date(self, year: int, quarter: str) -> datetime:
        """Get end date for specified quarter"""
        quarter_ends = {
            'q1': f"{year}-03-31",
            'q2': f"{year}-06-30", 
            'q3': f"{year}-09-30",
            'q4': f"{year}-12-31"
        }
        return datetime.strptime(quarter_ends[quarter], "%Y-%m-%d")
    
    async def _model_exists(self, version: str) -> bool:
        """Check if model version already exists"""
        query = f"""
        SELECT COUNT(*) as count
        FROM `{self.config.bigquery.model_registry_table}`
        WHERE version = '{version}' AND status = 'active'
        """
        result = await self.bigquery_client.query_to_dataframe(query)
        return result['count'].iloc[0] > 0
    
    async def _get_unprocessed_incidents(self, tech_centers: List[str]) -> pd.DataFrame:
        """Get incidents that haven't been processed for prediction"""
        batch_size = self.config.prediction.batch_size
        
        tech_center_filter = ','.join([f"'{tc}'" for tc in tech_centers])
        
        query = f"""
        SELECT *
        FROM `{self.config.bigquery.tables['raw_incidents']}` r
        LEFT JOIN `{self.config.bigquery.predictions_table}` p
        ON r.incident_id = p.incident_id
        WHERE r.tech_center IN ({tech_center_filter})
        AND p.incident_id IS NULL
        AND r.created_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        ORDER BY r.created_date DESC
        LIMIT {batch_size}
        """
        
        return await self.bigquery_client.query_to_dataframe(query)
    
    async def _store_predictions(self, predictions: List[Dict]) -> Dict:
        """Store prediction results in BigQuery"""
        if not predictions:
            return {"rows_inserted": 0}
        
        table_id = self.config.bigquery.predictions_table
        return await self.bigquery_client.insert_predictions(table_id, predictions)
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status and statistics"""
        return {
            "orchestrator_config": {
                "training_frequency": self.config.training.frequency,
                "training_window_months": self.config.training.training_window_months,
                "prediction_frequency_minutes": self.config.prediction.frequency_minutes,
                "tech_centers_count": len(self.config.tech_centers)
            },
            "current_models": self.current_models,
            "pipeline_stats": self.pipeline_stats,
            "last_updated": datetime.now().isoformat()
        }