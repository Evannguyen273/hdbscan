#!/usr/bin/env python3
"""
Enhanced HDBSCAN Clustering Pipeline with Tech Center Support
Supports preprocessing, training, and prediction pipelines
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import error handling system
from utils.error_handler import with_comprehensive_logging, catch_errors, PipelineLogger

from config.config import load_config
from pipeline.preprocessing_pipeline import PreprocessingPipeline
from pipeline.training_pipeline import TechCenterTrainingPipeline  
from pipeline.prediction_pipeline import PredictionPipeline
from pipeline.orchestrator import PipelineOrchestrator

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
    
    logging.info("Pipeline logging initialized")

@catch_errors
def run_preprocessing_pipeline(config, tech_center=None):
    """Run preprocessing pipeline with error handling"""
    pipeline = PreprocessingPipeline(config)
    if tech_center:
        result = pipeline.run_preprocessing_for_tech_center(tech_center)
    else:
        result = pipeline.run_preprocessing_all_tech_centers()
    return result

@catch_errors  
def run_training_pipeline(config, tech_center=None, year=None, quarter=None):
    """Run training pipeline with error handling"""
    pipeline = TechCenterTrainingPipeline(config)
    if tech_center:
        result = pipeline.train_single_tech_center(tech_center, year, quarter)
    else:
        result = pipeline.train_all_tech_centers_parallel(year, quarter)
    return result

@catch_errors
def run_prediction_pipeline(config, tech_center=None):
    """Run prediction pipeline with error handling"""
    pipeline = PredictionPipeline(config)
    if tech_center:
        result = pipeline.run_prediction_for_tech_center(tech_center)
    else:
        result = pipeline.run_predictions_all_tech_centers()
    return result

@catch_errors
def run_pipeline_scheduler(config_path):
    """Run automated scheduler with error handling"""
    orchestrator = PipelineOrchestrator(config_path)
    orchestrator.run_scheduler()

def main():
    parser = argparse.ArgumentParser(description="HDBSCAN Clustering Pipeline for Tech Centers")
    
    # Global options
    parser.add_argument("--config", default="config/enhanced_config.yaml", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Pipeline selection
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")
    
    # Preprocessing pipeline
    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing pipeline")
    preprocess_parser.add_argument("--tech-center", help="Process specific tech center only")
    
    # Training pipeline
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--tech-center", help="Train specific tech center only")
    train_parser.add_argument("--year", type=int, help="Training year (default: current year)")
    train_parser.add_argument("--quarter", choices=["q1", "q2", "q3", "q4"], help="Training quarter (default: current quarter)")
    train_parser.add_argument("--parallel", action="store_true", default=True, help="Enable parallel training")
    
    # Prediction pipeline
    predict_parser = subparsers.add_parser("predict", help="Run prediction pipeline")
    predict_parser.add_argument("--tech-center", help="Predict for specific tech center only")
    
    # Scheduler
    schedule_parser = subparsers.add_parser("schedule", help="Run automated scheduler")
    
    # Status check
    status_parser = subparsers.add_parser("status", help="Check pipeline status")
    
    # Legacy support for original pipeline
    legacy_parser = subparsers.add_parser("legacy", help="Run original clustering pipeline")
    legacy_parser.add_argument("--query", required=True, help="BigQuery query for data")
    legacy_parser.add_argument("--dataset", required=True, help="Dataset name")
    legacy_parser.add_argument("--embeddings-table", help="BigQuery table for embeddings")
    legacy_parser.add_argument("--results-table", help="BigQuery table for results")
    legacy_parser.add_argument("--start-stage", type=int, default=1, choices=[1,2,3,4], help="Start from stage")
    legacy_parser.add_argument("--end-stage", type=int, default=4, choices=[1,2,3,4], help="End at stage")
    legacy_parser.add_argument("--embedding-path", help="Path to precomputed embeddings")
    legacy_parser.add_argument("--summary-path", help="Path to precomputed summaries")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
      # Execute based on command
    try:
        if args.command == "preprocess":
            result = run_preprocessing_pipeline(config, args.tech_center)
            print(f"Preprocessing completed: {result.get('total_processed', 0)} incidents processed")
        
        elif args.command == "train":
            result = run_training_pipeline(config, args.tech_center, args.year, args.quarter)
            if args.tech_center:
                print(f"Training completed for {args.tech_center}: {result.get('status', 'Unknown')}")
            else:
                print(f"Training completed: {result.get('successful', 0)} successful, {result.get('failed', 0)} failed")
        
        elif args.command == "predict":
            result = run_prediction_pipeline(config, args.tech_center)
            if args.tech_center:
                print(f"Prediction completed for {args.tech_center}: {result.get('predicted_count', 0)} incidents")
            else:
                print(f"Predictions completed: {result.get('total_predicted', 0)} incidents predicted")
        
        elif args.command == "schedule":
            run_pipeline_scheduler(args.config)
        
        elif args.command == "status":
            orchestrator = PipelineOrchestrator(args.config)
            status = orchestrator.get_pipeline_status()
            
            print("=== PIPELINE STATUS ===")
            print(f"Timestamp: {status['timestamp']}")
            print(f"Tech Centers: {status['tech_centers']['total']}")
            print(f"Preprocessing: {status['preprocessing']['frequency']}")
            print(f"Prediction: {status['prediction']['frequency']}")
            print(f"Training: {status['training']['frequency']}")
            print(f"Save to Local: {status['configuration']['save_to_local']}")
            print(f"Result Path: {status['configuration']['result_path']}")
        
        elif args.command == "legacy":
            # Import legacy pipeline
            from pipeline import ClusteringPipeline
            
            pipeline = ClusteringPipeline(args.config)
            result = pipeline.run_modular_pipeline(
                input_query=args.query,
                embeddings_table_id=args.embeddings_table,
                results_table_id=args.results_table,
                dataset_name=args.dataset,
                embedding_path=args.embedding_path,
                summary_path=args.summary_path,
                start_from_stage=args.start_stage,
                end_at_stage=args.end_stage
            )
            
            print(f"Legacy pipeline completed for dataset: {args.dataset}")
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        print(f"❌ Pipeline execution failed: {e}")
        print("Check the logs for detailed error information.")
        sys.exit(1)

if __name__ == "__main__":
    main()