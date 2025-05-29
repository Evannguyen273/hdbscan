# Example: How to apply comprehensive logging to your pipeline modules

from utils.error_handler import with_comprehensive_logging, catch_errors, PipelineLogger

# Apply to preprocessing pipeline
@with_comprehensive_logging("preprocessing")
class PreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = PipelineLogger("preprocessing")
    
    def run_preprocessing_all_tech_centers(self):
        """Run preprocessing for all tech centers"""
        self.logger.log_stage_start("preprocessing_all", {"tech_centers": len(self.config.pipeline.tech_centers)})
        
        results = {"total_processed": 0, "successful_centers": 0, "failed_centers": 0}
        
        for i, tech_center in enumerate(self.config.pipeline.tech_centers):
            try:
                self.logger.log_progress(i + 1, len(self.config.pipeline.tech_centers), "tech centers")
                
                # Your preprocessing logic here
                center_result = self.run_preprocessing_for_tech_center(tech_center)
                results["total_processed"] += center_result.get("processed_count", 0)
                results["successful_centers"] += 1
                
            except Exception as e:
                self.logger.log_error("tech_center_processing", e, {"tech_center": tech_center})
                results["failed_centers"] += 1
                # Continue with other tech centers
        
        self.logger.log_stage_complete("preprocessing_all", results)
        return results

# Apply to training pipeline  
@with_comprehensive_logging("training")
class TechCenterTrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = PipelineLogger("training")
    
    def train_all_tech_centers_parallel(self, year=None, quarter=None):
        """Train models for all tech centers in parallel"""
        context = {"year": year, "quarter": quarter, "parallel": True}
        self.logger.log_stage_start("parallel_training", context)
        
        # Your training logic here
        results = {"successful": 0, "failed": 0, "models_created": []}
        
        self.logger.log_stage_complete("parallel_training", results)
        return results

# Apply to prediction pipeline
@with_comprehensive_logging("prediction") 
class PredictionPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = PipelineLogger("prediction")
    
    def run_predictions_all_tech_centers(self):
        """Run predictions for all tech centers"""
        self.logger.log_stage_start("predictions_all", {"tech_centers": len(self.config.pipeline.tech_centers)})
        
        # Your prediction logic here
        results = {"total_predicted": 0, "successful_centers": 0}
        
        self.logger.log_stage_complete("predictions_all", results)
        return results

# Apply to orchestrator
@catch_errors
class PipelineOrchestrator:
    def __init__(self, config_path):
        self.config_path = config_path
        self.logger = PipelineLogger("orchestrator")
    
    def run_scheduler(self):
        """Run the automated scheduler"""
        self.logger.log_stage_start("scheduler", {"mode": "continuous"})
        
        # Your scheduler logic here
        # This will automatically catch and log any errors
        
        self.logger.log_stage_complete("scheduler", {"status": "running"})

# Example usage in main functions
@catch_errors
def example_pipeline_function():
    """Example of how to use simple error catching"""
    print("Starting example pipeline...")
    
    # Simulate some work
    import time
    time.sleep(1)
    
    # Simulate potential error (uncomment to test)
    # raise Exception("Example error for testing")
    
    print("Example pipeline completed!")
    return {"status": "success", "processed": 100}

if __name__ == "__main__":
    # Test the error handling
    try:
        result = example_pipeline_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Caught error: {e}")