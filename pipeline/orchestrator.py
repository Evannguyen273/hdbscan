import logging
import schedule
import time
from datetime import datetime
from typing import Dict

from config.config import load_config
from pipeline.preprocessing_pipeline import PreprocessingPipeline
from pipeline.training_pipeline import TechCenterTrainingPipeline
from pipeline.prediction_pipeline import PredictionPipeline

class PipelineOrchestrator:
    """
    Main orchestrator for all pipeline operations:
    - Preprocessing: Every 1 hour
    - Prediction: Every 2 hours
    - Training: Quarterly (triggered manually or via schedule)
    """
    
    def __init__(self, config_path: str = "config/enhanced_config.yaml"):
        self.config = load_config(config_path)
        
        # Initialize pipelines
        self.preprocessing_pipeline = PreprocessingPipeline(self.config)
        self.training_pipeline = TechCenterTrainingPipeline(self.config)
        self.prediction_pipeline = PredictionPipeline(self.config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pipeline_orchestrator.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_preprocessing(self):
        """Run preprocessing pipeline"""
        self.logger.info("=== STARTING PREPROCESSING PIPELINE ===")
        try:
            result = self.preprocessing_pipeline.run_preprocessing_all_tech_centers()
            self.logger.info(f"Preprocessing completed: {result['total_processed']} incidents processed")
            return result
        except Exception as e:
            self.logger.error(f"Preprocessing pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_predictions(self):
        """Run prediction pipeline"""
        self.logger.info("=== STARTING PREDICTION PIPELINE ===")
        try:
            result = self.prediction_pipeline.run_predictions_all_tech_centers()
            self.logger.info(f"Predictions completed: {result['total_predicted']} incidents predicted")
            return result
        except Exception as e:
            self.logger.error(f"Prediction pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_training(self, year: int = None, quarter: str = None):
        """Run training pipeline for all tech centers"""
        self.logger.info("=== STARTING TRAINING PIPELINE ===")
        try:
            result = self.training_pipeline.train_all_tech_centers_parallel(year, quarter)
            self.logger.info(f"Training completed: {result['successful']} successful, {result['failed']} failed")
            return result
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_training_single_tech_center(self, tech_center: str, year: int = None, quarter: str = None):
        """Run training for a single tech center"""
        self.logger.info(f"=== STARTING TRAINING FOR {tech_center} ===")
        try:
            result = self.training_pipeline.train_single_tech_center(tech_center, year, quarter)
            self.logger.info(f"Training completed for {tech_center}: {result['status']}")
            return result
        except Exception as e:
            self.logger.error(f"Training failed for {tech_center}: {e}")
            return {"status": "failed", "error": str(e)}
    
    def setup_scheduled_jobs(self):
        """Setup scheduled jobs for automated pipeline execution"""
        
        # Preprocessing every hour
        schedule.every().hour.at(":00").do(self.run_preprocessing)
        
        # Predictions every 2 hours
        schedule.every(2).hours.do(self.run_predictions)
        
        # Optional: Quarterly training (uncomment if you want automatic quarterly training)
        # schedule.every().quarter.do(self.run_training)
        
        self.logger.info("Scheduled jobs configured:")
        self.logger.info("- Preprocessing: Every hour")
        self.logger.info("- Predictions: Every 2 hours")
        self.logger.info("- Training: Manual trigger (quarterly)")
    
    def run_scheduler(self):
        """Run the scheduler continuously"""
        self.setup_scheduled_jobs()
        self.logger.info("Pipeline scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("Pipeline scheduler stopped by user")
    
    def run_manual_job(self, job_type: str, **kwargs):
        """Run a pipeline job manually"""
        
        if job_type == "preprocessing":
            return self.run_preprocessing()
        
        elif job_type == "prediction":
            return self.run_predictions()
        
        elif job_type == "training":
            year = kwargs.get("year")
            quarter = kwargs.get("quarter")
            tech_center = kwargs.get("tech_center")
            
            if tech_center:
                return self.run_training_single_tech_center(tech_center, year, quarter)
            else:
                return self.run_training(year, quarter)
        
        else:
            raise ValueError(f"Unknown job type: {job_type}")
    
    def get_pipeline_status(self) -> Dict:
        """Get status of all pipelines"""
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "preprocessing": {
                "frequency": f"Every {self.config.pipeline.preprocessing.frequency_minutes} minutes",
                "batch_size": self.config.pipeline.preprocessing.batch_size
            },
            "prediction": {
                "frequency": f"Every {self.config.pipeline.prediction.frequency_minutes} minutes", 
                "batch_size": self.config.pipeline.prediction.batch_size
            },
            "training": {
                "frequency": "Quarterly (manual)",
                "parallel_training": self.config.pipeline.parallel_training,
                "max_workers": self.config.pipeline.max_workers
            },
            "tech_centers": {
                "total": len(self.config.pipeline.tech_centers),
                "list": self.config.pipeline.tech_centers
            },
            "configuration": {
                "save_to_local": self.config.pipeline.save_to_local,
                "result_path": self.config.pipeline.result_path,
                "use_checkpointing": self.config.pipeline.use_checkpointing
            }
        }
        
        return status