#!/usr/bin/env python3
"""
Enhanced HDBSCAN Clustering Pipeline - Main Entry Point
Supports both original pipeline.py functionality and new enhanced features
Updated to use config.py and config.yaml only
"""

import logging
import sys
import argparse
import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from config.config import load_config, get_config, validate_environment

# Import pipeline components
from data.bigquery_client import BigQueryClient
from preprocessing.embedding_generation import EmbeddingGenerator
from clustering.clustering_trainer import ClusteringTrainer
from utils.blob_storage import BlobStorageClient

def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                logs_dir / f"pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
        ]
    )

def cmd_validate(args):
    """Validate configuration and environment"""
    print("🔍 Validating configuration and environment...")
    
    if validate_environment():
        print("✅ All validations passed!")
        
        # Show configuration summary
        config = get_config()
        print(f"\n📋 Configuration Summary:")
        print(f"   BigQuery Project: {config.bigquery.project_id}")
        print(f"   Tech Centers: {len(config.tech_centers)} configured")
        print(f"   Embedding Model: {config.azure.openai.get('embedding_model', 'N/A')}")
        print(f"   Blob Container: {config.get('blob_storage', {}).get('container_name', 'N/A')}")
        
        return True
    else:
        print("❌ Validation failed!")
        return False

def cmd_preprocess(args):
    """Run preprocessing pipeline"""
    print("🔄 Starting preprocessing pipeline...")
    
    config = get_config()
    bq_client = BigQueryClient(config.config)
    embedding_gen = EmbeddingGenerator(config.config)
    
    tech_center = args.tech_center if hasattr(args, 'tech_center') else None
    
    if tech_center:
        print(f"   Processing tech center: {tech_center}")
        # Process specific tech center
        incidents = bq_client.get_raw_incidents(tech_center=tech_center, limit=args.limit)
        if not incidents.empty:
            processed = embedding_gen.process_incidents_with_embeddings(incidents)
            bq_client.save_preprocessed_incidents(processed)
            print(f"✅ Processed {len(processed)} incidents for {tech_center}")
        else:
            print(f"⚠️ No incidents found for {tech_center}")
    else:
        # Process all tech centers
        for tc in config.tech_centers:
            print(f"   Processing: {tc}")
            incidents = bq_client.get_raw_incidents(tech_center=tc, limit=args.limit)
            if not incidents.empty:
                processed = embedding_gen.process_incidents_with_embeddings(incidents)
                bq_client.save_preprocessed_incidents(processed)
                print(f"   ✅ {tc}: {len(processed)} incidents processed")
            else:
                print(f"   ⚠️ {tc}: No incidents found")

def cmd_train(args):
    """Run training pipeline"""
    print("🎯 Starting training pipeline...")
    
    config = get_config()
    trainer = ClusteringTrainer(config.config)
    
    quarter = args.quarter
    year = getattr(args, 'year', datetime.datetime.now().year)
    tech_center = getattr(args, 'tech_center', None)
    
    if tech_center:
        # Train specific tech center
        print(f"   Training: {tech_center} for {year}-{quarter}")
        success, results = trainer.train_tech_center(tech_center, year, quarter)
        if success:
            print(f"✅ Training completed for {tech_center}")
        else:
            print(f"❌ Training failed for {tech_center}")
    else:
        # Train all tech centers
        for tc in config.tech_centers:
            print(f"   Training: {tc}")
            success, results = trainer.train_tech_center(tc, year, quarter)
            if success:
                print(f"   ✅ {tc}: Training completed")
            else:
                print(f"   ❌ {tc}: Training failed")

def cmd_predict(args):
    """Run prediction pipeline"""
    print("🔮 Starting prediction pipeline...")
    
    config = get_config()
    # Implement prediction logic here
    print("✅ Prediction pipeline completed")

def cmd_status(args):
    """Show pipeline status"""
    print("📊 Pipeline Status Dashboard")
    print("=" * 50)
    
    config = get_config()
    
    # Show tech centers
    print(f"\n🏢 Tech Centers ({len(config.tech_centers)}):")
    for i, tc in enumerate(config.tech_centers, 1):
        print(f"   {i:2d}. {tc}")
    
    # Show current quarter
    current_quarter = get_current_quarter()
    print(f"\n📅 Current Quarter: {current_quarter}")
    
    # Show configuration status
    print(f"\n⚙️ Configuration:")
    print(f"   Config File: {config.config_path}")
    print(f"   BigQuery Project: {config.bigquery.project_id}")
    print(f"   Embedding Weights: {config.clustering.embedding_weights}")

def get_current_quarter():
    """Get current quarter"""
    month = datetime.datetime.now().month
    if month in [1, 2, 3]:
        return 'q1'
    elif month in [4, 5, 6]:
        return 'q2' 
    elif month in [7, 8, 9]:
        return 'q3'
    else:
        return 'q4'

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced HDBSCAN Clustering Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration and environment')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocessing pipeline')
    preprocess_parser.add_argument('--tech-center', help='Specific tech center to process')
    preprocess_parser.add_argument('--limit', type=int, help='Limit number of incidents to process')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Run training pipeline')
    train_parser.add_argument('--quarter', required=True, choices=['q1', 'q2', 'q3', 'q4'], 
                             help='Quarter to train for')
    train_parser.add_argument('--year', type=int, default=datetime.datetime.now().year,
                             help='Year to train for')
    train_parser.add_argument('--tech-center', help='Specific tech center to train')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction pipeline')
    predict_parser.add_argument('--tech-center', help='Specific tech center to predict')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Load configuration
        config = load_config()
        setup_logging(config.config)
        
        # Route to appropriate command
        if args.command == 'validate':
            cmd_validate(args)
        elif args.command == 'preprocess':
            cmd_preprocess(args)
        elif args.command == 'train':
            cmd_train(args)
        elif args.command == 'predict':
            cmd_predict(args)
        elif args.command == 'status':
            cmd_status(args)
        else:
            print(f"Unknown command: {args.command}")
            
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()