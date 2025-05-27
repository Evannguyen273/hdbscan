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

from config.config import load_config
from pipeline.preprocessing_pipeline import PreprocessingPipeline
from pipeline.training_pipeline import TechCenterTrainingPipeline  
from pipeline.prediction_pipeline import PredictionPipeline
from pipeline.orchestrator import PipelineOrchestrator

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )

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
            pipeline = PreprocessingPipeline(config)
            if args.tech_center:
                result = pipeline.run_preprocessing_for_tech_center(args.tech_center)
            else:
                result = pipeline.run_preprocessing_all_tech_centers()
            
            print(f"Preprocessing completed: {result['total_processed']} incidents processed")
        
        elif args.command == "train":
            pipeline = TechCenterTrainingPipeline(config)
            if args.tech_center:
                result = pipeline.train_single_tech_center(args.tech_center, args.year, args.quarter)
                print(f"Training completed for {args.tech_center}: {result['status']}")
            else:
                result = pipeline.train_all_tech_centers_parallel(args.year, args.quarter)
                print(f"Training completed: {result['successful']} successful, {result['failed']} failed")
        
        elif args.command == "predict":
            pipeline = PredictionPipeline(config)
            if args.tech_center:
                result = pipeline.run_prediction_for_tech_center(args.tech_center)
                print(f"Prediction completed for {args.tech_center}: {result['predicted_count']} incidents")
            else:
                result = pipeline.run_predictions_all_tech_centers()
                print(f"Predictions completed: {result['total_predicted']} incidents predicted")
        
        elif args.command == "schedule":
            orchestrator = PipelineOrchestrator(args.config)
            orchestrator.run_scheduler()
        
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
        sys.exit(1)

if __name__ == "__main__":
    main()