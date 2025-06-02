#!/usr/bin/env python3
"""
Enhanced HDBSCAN Clustering Pipeline with Hybrid Domain Grouping
Integrates the spatial + semantic domain grouping approach from Untitled-1
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Local imports
from training_pipeline import EnhancedTrainingPipeline
from config.config import get_config
from logging_setup import setup_detailed_logging


def setup_logging():
    """Setup logging for the main pipeline"""
    log_file = setup_detailed_logging(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"HDBSCAN Pipeline started at {datetime.now()}")
    logger.info(f"Detailed logs: {log_file}")
    return logger


def train_command(args):
    """Execute training command with cumulative approach and versioned tables"""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize enhanced training pipeline
        pipeline = EnhancedTrainingPipeline()
        
        # Get tech centers to train
        tech_centers = args.tech_centers if args.tech_centers else [
            "BT-TC-Product Development & Engineering",
            "BT-TC-Network Operations", 
            "BT-TC-Data Analytics",
            "BT-TC-DevOps Engineering",
            "BT-TC-Business Intelligence"
        ]
        
        quarters = args.quarters if args.quarters else ["q4"]
        year = args.year
        
        logger.info(f"Starting cumulative training for {len(tech_centers)} tech centers, quarters: {quarters}")
        logger.info(f"Training approach: 24-month rolling window with versioned tables")
        
        results_summary = {}
        
        for tech_center in tech_centers:
            for quarter in quarters:
                try:
                    logger.info(f"Training {tech_center} - Q{quarter} {year}")
                    
                    # Run enhanced training with cumulative approach and domain grouping
                    results = pipeline.run_training_with_domains(
                        tech_center=tech_center,
                        quarter=quarter,
                        year=year
                    )
                    
                    # Extract versioned table information
                    table_info = results["summary"]["bigquery_table"]
                    training_window = results["summary"]["training_window"]
                    
                    results_summary[f"{tech_center}_{quarter}"] = {
                        "status": "success",
                        "domains_count": results["summary"]["domains_count"],
                        "clusters_count": results["summary"]["clusters_count"],
                        "table_name": table_info["table_name"],
                        "model_version": table_info["model_version"],
                        "record_count": table_info["record_count"],
                        "training_window": training_window,
                        "output_dir": results["output_dir"]
                    }
                    
                    print(f"✅ {tech_center} Q{quarter}:")
                    print(f"   Table: {table_info['table_name']}")
                    print(f"   Domains: {results['summary']['domains_count']}")
                    print(f"   Window: {training_window['start_date']} to {training_window['end_date']}")
                    print(f"   Records: {table_info['record_count']}")
                    
                except Exception as e:
                    logger.error(f"Training failed for {tech_center} Q{quarter}: {e}")
                    results_summary[f"{tech_center}_{quarter}"] = {
                        "status": "failed",
                        "error": str(e)
                    }
                    print(f"❌ {tech_center} Q{quarter}: Failed - {e}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print("CUMULATIVE TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Training Approach: 24-month rolling window")
        print(f"Table Strategy: Versioned tables per training cycle")
        print()
        
        successful = sum(1 for r in results_summary.values() if r["status"] == "success")
        total = len(results_summary)
        
        print(f"Successful: {successful}/{total}")
        print(f"Failed: {total - successful}/{total}")
        
        if successful > 0:
            print(f"\n✅ Cumulative training with versioned tables completed!")
            print(f"📊 Created {successful} versioned BigQuery tables")
            print(f"📁 Results saved to: results/")
            print(f"\n📋 Created Tables:")
            for key, result in results_summary.items():
                if result["status"] == "success":
                    print(f"   • {result['table_name']} ({result['record_count']} records)")
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Training command failed: {e}")
        print(f"❌ Training failed: {e}")
        return None


def predict_command(args):
    """Execute prediction command (placeholder)"""
    logger = logging.getLogger(__name__)
    logger.info("Prediction command called")
    
    print("🔮 Prediction functionality coming soon!")
    print("   This will generate predictions using trained models with domain mappings")
    
    return {"status": "not_implemented"}


def status_command(args):
    """Show pipeline status"""
    logger = logging.getLogger(__name__)
    logger.info("Status command called")
    
    # Check results directory
    results_path = Path("results")
    if not results_path.exists():
        print("📊 No results directory found")
        return
    
    # List training results
    training_dirs = list(results_path.glob("*_*_q*"))
    
    print(f"📊 PIPELINE STATUS")
    print(f"{'='*40}")
    print(f"Results directory: {results_path.absolute()}")
    print(f"Training runs found: {len(training_dirs)}")
    
    if training_dirs:
        print(f"\nRecent training runs:")
        for training_dir in sorted(training_dirs, reverse=True)[:5]:
            # Check for domain results
            domain_file = training_dir / "domains.json"
            summary_file = training_dir / "training_summary.json"
            
            status = "✅ Complete" if domain_file.exists() and summary_file.exists() else "⚠️ Incomplete"
            print(f"  {training_dir.name}: {status}")
    
    return {"results_count": len(training_dirs)}


def preprocess_command(args):
    """Execute preprocessing command (placeholder)"""
    logger = logging.getLogger(__name__)
    logger.info("Preprocessing command called")
    
    print("🔄 Preprocessing functionality coming soon!")
    print("   This will preprocess incident data and generate embeddings")
    
    return {"status": "not_implemented"}


def print_storage_optimization_info():
    """Print information about storage cost optimization and versioned tables"""
    print(f"\n{'='*60}")
    print(f"💾 STORAGE ARCHITECTURE & MODEL VERSIONING")
    print(f"{'='*60}")
    print("📊 Complete Storage Strategy:")
    print()
    print("📁 preprocessed_incidents (BigQuery)")
    print("   └── Contains: embeddings, combined_incidents_summary")
    print("   └── Storage: HIGH (1536-dim embeddings)")
    print("   └── Usage: Source data for training/prediction")
    print("   └── Retention: 24-month rolling window")
    print()
    print("🗂️  Model Artifacts (Azure Blob Storage)")
    print("   └── Path: hdbscan-models/{tech_center}/{year}_{quarter}/")
    print("   └── Contains: umap_model.pkl, hdbscan_model.pkl, embeddings.npy")
    print("   └── Storage: MEDIUM (trained models + embeddings)")
    print("   └── Usage: Production prediction pipeline")
    print("   └── Versioning: Separate folder per training cycle")
    print("   └── Examples:")
    print("      • hdbscan-models/bt-tc-data-analytics/2025_q2/")
    print("      • hdbscan-models/bt-tc-network-operations/2025_q2/")
    print("      • hdbscan-models/bt-tc-data-analytics/2025_q4/")
    print()
    print("📁 clustering_predictions_{year}_{quarter}_{hash} (BigQuery)")
    print("   └── Contains: cluster_id, domain_id, umap_x, umap_y")
    print("   └── Storage: LOW (no embeddings, only coordinates)")
    print("   └── Usage: Training results + domain mappings")
    print("   └── Versioning: New table per training cycle")
    print("   └── Reference: Points to blob storage model artifacts")
    print()
    print("📁 incident_predictions (BigQuery)")
    print("   └── Contains: predicted_cluster_id, confidence_score")
    print("   └── Storage: LOW (no embeddings, only predictions)")
    print("   └── Usage: Real-time classification results")
    print("   └── Reference: model_table_used, blob_model_path")
    print()
    print("🔄 Complete Workflow:")
    print("   1. Training: Load data from preprocessed_incidents")
    print("   2. Training: Save models to blob storage")
    print("   3. Training: Save results to versioned BigQuery table")
    print("   4. Prediction: Load models from blob storage")
    print("   5. Prediction: Load domain mappings from BigQuery")
    print("   6. Prediction: Save results to incident_predictions")
    print()
    print("💰 Cost Optimization:")
    print("   • BigQuery: ~50% storage reduction (no duplicate embeddings)")
    print("   • Blob Storage: Efficient model artifact storage")
    print("   • Separation: Training data vs. Model artifacts vs. Results")
    print()
    print("📋 Model Versioning Benefits:")
    print("   • Clear artifact lineage in blob storage")
    print("   • Easy rollback to previous model versions")
    print("   • Parallel model testing (A/B testing)")
    print("   • Cost-effective long-term model retention")
    print(f"{'='*60}")
    print()
    print("Example Storage Layout:")
    print("  Azure Blob Storage:")
    print("    hdbscan-models/")
    print("    ├── bt-tc-data-analytics/")
    print("    │   ├── 2024_q4/  ← Previous model")
    print("    │   │   ├── umap_model.pkl")
    print("    │   │   ├── hdbscan_model.pkl")
    print("    │   │   └── model_metadata.json")
    print("    │   └── 2025_q2/  ← Current model")
    print("    │       ├── umap_model.pkl")
    print("    │       ├── hdbscan_model.pkl")
    print("    │       └── model_metadata.json")
    print("    └── bt-tc-network-operations/")
    print("        └── 2025_q2/")
    print("            └── ...")
    print()
    print("  BigQuery Tables:")
    print("    • clustering_predictions_2024_q4_789  ← Previous results")
    print("    • clustering_predictions_2025_q2_789  ← Current results")
    print("    • incident_predictions                ← Live predictions")
    print(f"{'='*60}")


def main():
    """Main entry point for the enhanced HDBSCAN pipeline"""
    parser = argparse.ArgumentParser(
        description="Enhanced HDBSCAN Clustering Pipeline with Hybrid Domain Grouping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --quarter q4 --year 2024
  python main.py train --tech-centers "BT-TC-Data Analytics" --quarter q3 q4
  python main.py status
  python main.py predict
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train clustering models with domain grouping')
    train_parser.add_argument('--tech-centers', nargs='+', help='Tech centers to train')
    train_parser.add_argument('--quarters', nargs='+', default=['q4'], help='Quarters to train (q1, q2, q3, q4)')
    train_parser.add_argument('--year', type=int, default=2024, help='Year for training data')
    train_parser.set_defaults(func=train_command)
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions using trained models')
    predict_parser.add_argument('--tech-centers', nargs='+', help='Tech centers to predict')
    predict_parser.add_argument('--quarter', default='q4', help='Quarter for predictions')
    predict_parser.add_argument('--year', type=int, default=2024, help='Year for predictions')
    predict_parser.set_defaults(func=predict_command)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    status_parser.set_defaults(func=status_command)
    
    # Storage info command  
    info_parser = subparsers.add_parser('info', help='Show storage optimization information')
    info_parser.set_defaults(func=lambda args: print_storage_optimization_info())
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess incident data')
    preprocess_parser.add_argument('--tech-centers', nargs='+', help='Tech centers to preprocess')
    preprocess_parser.add_argument('--quarter', default='q4', help='Quarter to preprocess')
    preprocess_parser.add_argument('--year', type=int, default=2024, help='Year to preprocess')
    preprocess_parser.set_defaults(func=preprocess_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            result = args.func(args)
            if result:
                logger.info(f"Command '{args.command}' completed successfully")
            else:
                logger.warning(f"Command '{args.command}' completed with issues")
        except Exception as e:
            logger.error(f"Command '{args.command}' failed: {e}")
            print(f"❌ Command failed: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        print(f"\n💡 Try: python main.py train --quarter q4")
        print(f"     or: python main.py status")


if __name__ == "__main__":
    main()