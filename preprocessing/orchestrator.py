# preprocessing/orchestrator.py
# Updated for new config structure and cumulative training approach
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datetime import datetime

from .text_processing import TextProcessor
from .embedding_processor import EmbeddingProcessor
from config.config import get_config

class PreprocessingOrchestrator:
    """Orchestrates the complete preprocessing pipeline with comprehensive error tracking"""
    
    def __init__(self, config=None):
        """Initialize preprocessing orchestrator with updated config system"""
        self.config = config if config is not None else get_config()
        self.text_processor = TextProcessor(self.config)
        self.embedding_processor = EmbeddingProcessor(self.config)
        
        # Track pipeline statistics
        self.pipeline_stats = {}
        
    def run_complete_pipeline(self, df: pd.DataFrame, 
                            summarization_batch_size: int = 10,
                            embedding_batch_size: int = 50,
                            use_batch_embedding_api: bool = True) -> Tuple[pd.Series, np.ndarray, pd.Index, Dict]:
        """
        Run the complete preprocessing pipeline: summarization -> embedding -> clustering preparation.
        
        Args:
            df: Input DataFrame with incident data
            summarization_batch_size: Batch size for summarization
            embedding_batch_size: Batch size for embedding
            use_batch_embedding_api: Whether to use batch embedding API
            
        Returns:
            Tuple of (summaries_series, embedding_matrix, valid_indices, comprehensive_stats)
        """
        pipeline_start_time = datetime.now()
        logging.info("Starting complete preprocessing pipeline for %d incidents", len(df))
        
        # Stage 1: Summarization
        logging.info("=" * 60)
        logging.info("STAGE 1: SUMMARIZATION")
        logging.info("=" * 60)
        
        summaries_series, summarization_stats = self.text_processor.process_incident_for_embedding_batch(
            df, batch_size=summarization_batch_size
        )
        
        # Stage 2: Embedding
        logging.info("=" * 60)
        logging.info("STAGE 2: EMBEDDING")
        logging.info("=" * 60)
        
        embeddings_series, embedding_stats = self.embedding_processor.process_embeddings_batch(
            summaries_series, 
            batch_size=embedding_batch_size,
            use_batch_api=use_batch_embedding_api
        )
        
        # Stage 3: Clustering Preparation
        logging.info("=" * 60)
        logging.info("STAGE 3: CLUSTERING PREPARATION")
        logging.info("=" * 60)
        
        embedding_matrix, valid_indices = self.embedding_processor.get_clustering_ready_data(embeddings_series)
        
        # Compile comprehensive statistics
        pipeline_duration = datetime.now() - pipeline_start_time
        
        comprehensive_stats = {
            "pipeline_info": {
                "start_time": pipeline_start_time.isoformat(),
                "duration_seconds": pipeline_duration.total_seconds(),
                "total_input_incidents": len(df)
            },
            "summarization": summarization_stats,
            "embedding": embedding_stats,
            "clustering_ready": {
                "incidents_with_embeddings": len(valid_indices),
                "embedding_dimensions": embedding_matrix.shape[1] if len(embedding_matrix) > 0 else 0,
                "ready_for_clustering": len(embedding_matrix) > 0
            },
            "overall_pipeline": self._calculate_overall_stats(summarization_stats, embedding_stats, len(df))
        }
        
        # Log final pipeline summary
        self._log_pipeline_summary(comprehensive_stats)
        
        # Store stats for later access
        self.pipeline_stats = comprehensive_stats
        
        return summaries_series, embedding_matrix, valid_indices, comprehensive_stats
    
    def _calculate_overall_stats(self, summarization_stats: Dict, embedding_stats: Dict, total_incidents: int) -> Dict:
        """Calculate overall pipeline statistics"""
        return {
            "total_incidents": total_incidents,
            "incidents_ready_for_clustering": embedding_stats.get("successful_embeddings", 0),
            "overall_success_rate": (embedding_stats.get("successful_embeddings", 0) / total_incidents) * 100 if total_incidents > 0 else 0,
            "summarization_success_rate": summarization_stats.get("success_rate", 0),
            "embedding_success_rate": embedding_stats.get("embedding_success_rate", 0),
            "total_failures": {
                "summarization_failures": summarization_stats.get("failed_incidents", 0),
                "embedding_failures": embedding_stats.get("embedding_failures", 0),
                "total_failed_incidents": summarization_stats.get("failed_incidents", 0) + embedding_stats.get("embedding_failures", 0)
            }
        }
    
    def _log_pipeline_summary(self, stats: Dict):
        """Log comprehensive pipeline summary"""
        logging.info("=" * 60)
        logging.info("PIPELINE COMPLETE - SUMMARY")
        logging.info("=" * 60)
        
        overall = stats["overall_pipeline"]
        clustering = stats["clustering_ready"]
        
        logging.info("Overall Results:")
        logging.info("  - Total incidents processed: %d", overall["total_incidents"])
        logging.info("  - Incidents ready for clustering: %d (%.1f%%)", 
                    clustering["incidents_with_embeddings"], overall["overall_success_rate"])
        
        logging.info("Stage Success Rates:")
        logging.info("  - Summarization: %.1f%%", overall["summarization_success_rate"])
        logging.info("  - Embedding: %.1f%%", overall["embedding_success_rate"])
        logging.info("  - Overall pipeline: %.1f%%", overall["overall_success_rate"])
        
        if overall["total_failures"]["total_failed_incidents"] > 0:
            logging.warning("Failure Summary:")
            logging.warning("  - Summarization failures: %d", overall["total_failures"]["summarization_failures"])
            logging.warning("  - Embedding failures: %d", overall["total_failures"]["embedding_failures"])
            logging.warning("  - Total failed incidents: %d", overall["total_failures"]["total_failed_incidents"])
        
        if clustering["ready_for_clustering"]:
            logging.info("Clustering Data Ready:")
            logging.info("  - Embedding matrix shape: (%d, %d)", 
                        clustering["incidents_with_embeddings"], clustering["embedding_dimensions"])
        else:
            logging.error("No data available for clustering - all incidents failed processing")
    
    def get_failed_incidents_comprehensive_report(self) -> Dict:
        """Get comprehensive report of all failures across the pipeline"""
        summarization_report = self.text_processor.get_failed_incidents_report()
        embedding_report = self.embedding_processor.get_failed_incidents_report()
        
        return {
            "summarization_failures": summarization_report,
            "embedding_failures": embedding_report,
            "pipeline_summary": {
                "total_summarization_failures": summarization_report["failed_incident_count"],
                "total_embedding_failures": embedding_report["embedding_failures"]["count"],
                "total_pipeline_failures": (
                    summarization_report["failed_incident_count"] + 
                    embedding_report["total_pipeline_failures"]
                ),
                "incidents_lost_at_summarization": list(set(summarization_report["failed_incident_numbers"])),
                "incidents_lost_at_embedding": list(set(embedding_report["embedding_failures"]["incident_numbers"])),
                "all_failed_incidents": list(set(
                    summarization_report["failed_incident_numbers"] + 
                    embedding_report["embedding_failures"]["incident_numbers"] +
                    embedding_report["summarization_failures"]["incident_numbers"]
                ))
            }
        }
    
    def get_pipeline_statistics(self) -> Dict:
        """Get the stored pipeline statistics from the last run"""
        return self.pipeline_stats