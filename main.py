"""
HDBSCAN Clustering Pipeline - Main Execution Script

This script provides a complete, modularized HDBSCAN clustering pipeline for IT incidents.
It supports stage-wise execution, checkpointing, and Azure OpenAI integration.

Usage examples:
    # Run complete pipeline
    python main.py --dataset incidents_2024 --start-stage 1 --end-stage 4
    
    # Run only embeddings generation
    python main.py --dataset incidents_2024 --start-stage 1 --end-stage 1
    
    # Resume from clustering stage
    python main.py --dataset incidents_2024 --start-stage 2 --end-stage 4
"""

import logging
import argparse
import sys
import os
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from main_pipeline import ClusteringPipeline

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"clustering_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="HDBSCAN Clustering Pipeline for IT Incidents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name for organizing outputs"
    )
    
    # Pipeline control
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Stage to start from (1=embeddings, 2=clustering, 3=analysis, 4=save)"
    )
    
    parser.add_argument(
        "--end-stage",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Stage to end at (1=embeddings, 2=clustering, 3=analysis, 4=save)"
    )
    
    # Data input
    parser.add_argument(
        "--query",
        help="BigQuery SQL query for data input (required for stage 1)"
    )
    
    parser.add_argument(
        "--embedding-path",
        help="Path to existing embeddings file (for stages 2+)"
    )
    
    parser.add_argument(
        "--summary-path",
        help="Path to existing summaries file (for stage 1)"
    )
    
    # BigQuery outputs
    parser.add_argument(
        "--embeddings-table",
        help="BigQuery table ID for saving embeddings"
    )
    
    parser.add_argument(
        "--results-table",
        help="BigQuery table ID for saving final results"
    )
    
    parser.add_argument(
        "--write-disposition",
        choices=["WRITE_APPEND", "WRITE_TRUNCATE"],
        default="WRITE_APPEND",
        help="BigQuery write disposition"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing (recompute everything)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate command line arguments"""
    
    # Stage validation
    if args.start_stage > args.end_stage:
        raise ValueError(f"Start stage ({args.start_stage}) cannot be greater than end stage ({args.end_stage})")
    
    # Query required for stage 1
    if args.start_stage == 1 and not args.query:
        raise ValueError("--query is required when starting from stage 1")
    
    # Dataset name validation
    if not args.dataset.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Dataset name must be alphanumeric (underscores and hyphens allowed)")
    
    return True

def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate arguments
        validate_arguments(args)
        
        # Log execution parameters
        logger.info("="*80)
        logger.info("HDBSCAN CLUSTERING PIPELINE STARTED")
        logger.info("="*80)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Stages: {args.start_stage} to {args.end_stage}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Checkpointing: {'Disabled' if args.no_checkpoint else 'Enabled'}")
        
        # Initialize pipeline
        logger.info("Initializing clustering pipeline...")
        pipeline = ClusteringPipeline(config_path=args.config)
        
        # Run pipeline
        results = pipeline.run_modular_pipeline(
            input_query=args.query,
            embeddings_table_id=args.embeddings_table,
            results_table_id=args.results_table,
            dataset_name=args.dataset,
            embedding_path=args.embedding_path,
            summary_path=args.summary_path,
            write_disposition=args.write_disposition,
            start_from_stage=args.start_stage,
            end_at_stage=args.end_stage,
            use_checkpoint=not args.no_checkpoint
        )
        
        # Log completion
        logger.info("="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        # Log stage results
        for stage_name, stage_result in results.items():
            if stage_name == "saved":
                logger.info(f"Results saved to BigQuery: {stage_result}")
            elif isinstance(stage_result, dict) and "df_with_embeddings" in stage_result:
                df = stage_result["df_with_embeddings"]
                logger.info(f"Embeddings generated: {len(df)} records")
            elif isinstance(stage_result, dict) and "clustered_df" in stage_result:
                df = stage_result["clustered_df"]
                clusters = df['cluster'].nunique()
                logger.info(f"Clustering completed: {len(df)} records, {clusters} clusters")
            elif isinstance(stage_result, dict) and "final_df" in stage_result:
                df = stage_result["final_df"]
                domains = len(stage_result.get("domains", {}).get("domains", []))
                logger.info(f"Analysis completed: {len(df)} records, {domains} domains")
        
        logger.info(f"Output files saved to: {pipeline.config.pipeline.result_path}/{args.dataset}/")
        
        return 0
        
    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)