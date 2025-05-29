import logging
import schedule
import time
import os
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
        """Run preprocessing pipeline for all tech centers"""
        try:
            self.logger.info("=== STARTING PREPROCESSING PIPELINE ===")
            
            # Run preprocessing for all tech centers
            results = self.preprocessing_pipeline.run_preprocessing_all_tech_centers()
            
            # Log comprehensive results with failed incidents
            self.logger.info(f"Preprocessing completed: {results['successful']}/{results['total_tech_centers']} tech centers successful")
            self.logger.info(f"Total incidents processed: {results['total_processed']}")
            
            if results.get('total_failed_incidents', 0) > 0:
                self.logger.warning(f"FAILED INCIDENTS: {results['total_failed_incidents']} incidents failed summarization")
                self.logger.warning(f"Failed incident numbers: {results.get('failed_incident_numbers', [])}")
                
                # Create alert for failed incidents
                self._create_failed_incidents_alert(results)
            
            self.logger.info("=== PREPROCESSING PIPELINE COMPLETED ===")
            return results
            
        except Exception as e:
            self.logger.error(f"Preprocessing pipeline failed: {e}")
            raise
    
    def _create_failed_incidents_alert(self, results: Dict):
        """Create an alert/notification for failed incidents that need investigation"""
        try:
            failed_incidents = results.get('failed_incident_numbers', [])
            
            if not failed_incidents:
                return
            
            alert_message = f"""
PREPROCESSING ALERT: Failed Incident Summarization

Summary:
- Total failed incidents: {len(failed_incidents)}
- Failed incident numbers: {', '.join(map(str, failed_incidents))}
- Timestamp: {results.get('timestamp', 'Unknown')}

Action Required:
1. Check incident details in ServiceNow for these numbers
2. Review error logs in preprocessing/failed_incidents/ directory
3. Determine if manual processing or data cleanup is needed

Tech Center Details:
"""
            
            for tech_result in results.get('results', []):
                if tech_result.get('failed_incidents'):
                    alert_message += f"- {tech_result['tech_center']}: {len(tech_result['failed_incidents'])} failed\n"
            
            # Log the alert
            self.logger.warning("=" * 60)
            self.logger.warning("FAILED INCIDENTS ALERT")
            self.logger.warning("=" * 60)
            self.logger.warning(alert_message)
            self.logger.warning("=" * 60)
            
            # Save alert to file for external monitoring
            if hasattr(self.config.pipeline, 'save_to_local') and self.config.pipeline.save_to_local:
                alert_dir = f"{self.config.pipeline.result_path}/preprocessing/alerts"
                os.makedirs(alert_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                alert_file = f"{alert_dir}/failed_incidents_alert_{timestamp}.txt"
                
                with open(alert_file, "w") as f:
                    f.write(alert_message)
                
                self.logger.info(f"Failed incidents alert saved to {alert_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to create alert for failed incidents: {e}")
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