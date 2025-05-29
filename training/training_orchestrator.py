# filepath: c:\Users\Bachn\OneDrive\Desktop\my_projects\hdbscan\training\training_orchestrator.py
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from datetime import datetime

from .clustering_trainer import ClusteringTrainer
from ..preprocessing.orchestrator import PreprocessingOrchestrator

class TrainingOrchestrator:
    """Orchestrates the complete training pipeline from raw data to trained clustering model"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessing_orchestrator = PreprocessingOrchestrator(config)
        self.clustering_trainer = ClusteringTrainer(config)
        
        # Track pipeline statistics
        self.pipeline_stats = {}
        
    def run_end_to_end_training(self, df: pd.DataFrame,
                              preprocessing_config: Dict = None,
                              training_config: Dict = None,
                              save_intermediate_results: bool = True) -> Tuple[bool, Dict]:
        """
        Run complete end-to-end training pipeline with comprehensive error handling.
        
        Args:
            df: Raw incident DataFrame
            preprocessing_config: Configuration for preprocessing stage
            training_config: Configuration for training stage
            save_intermediate_results: Whether to save intermediate results
            
        Returns:
            Tuple of (pipeline_successful, comprehensive_results)
        """
        pipeline_start_time = datetime.now()
        logging.info("=" * 80)
        logging.info("STARTING END-TO-END TRAINING PIPELINE")
        logging.info("=" * 80)
        logging.info("Input: %d raw incidents", len(df))
        
        # Initialize configurations
        preprocessing_config = preprocessing_config or {
            "summarization_batch_size": 20,
            "embedding_batch_size": 50,
            "use_batch_embedding_api": True
        }
        
        training_config = training_config or {
            "hdbscan_params": {
                "min_cluster_size": 15,
                "min_samples": 5,
                "cluster_selection_epsilon": 0.0,
                "metric": "euclidean"
            },
            "fallback_params_list": [
                {"min_cluster_size": 10, "min_samples": 3, "metric": "euclidean"},
                {"min_cluster_size": 8, "min_samples": 2, "metric": "cosine"},
                {"min_cluster_size": 5, "min_samples": 1, "metric": "euclidean"}
            ],
            "preprocessing_config": {"apply_scaling": True, "apply_pca": False},
            "output_dir": "models"
        }
        
        # Initialize results tracking
        results = {
            "pipeline_successful": False,
            "preprocessing_results": {},
            "training_results": {},
            "pipeline_stats": {},
            "critical_failures": [],
            "warnings": []
        }
        
        try:
            # Stage 1: Preprocessing Pipeline
            logging.info("STAGE 1: PREPROCESSING PIPELINE")
            logging.info("=" * 50)
            
            summaries, embedding_matrix, valid_indices, preprocessing_stats = self.preprocessing_orchestrator.run_complete_pipeline(
                df=df,
                **preprocessing_config
            )
            
            results["preprocessing_results"] = {
                "stats": preprocessing_stats,
                "summaries_count": len(summaries) if summaries is not None else 0,
                "embeddings_shape": embedding_matrix.shape if embedding_matrix is not None else (0, 0),
                "valid_indices_count": len(valid_indices) if valid_indices is not None else 0
            }
            
            # Check if preprocessing provided sufficient data for training
            if embedding_matrix is None or len(embedding_matrix) == 0:
                critical_error = "Preprocessing failed to produce any embeddings for training"
                results["critical_failures"].append(critical_error)
                logging.error("❌ %s", critical_error)
                return False, results            # Check minimum data requirements (more flexible)
            min_samples_for_clustering = self.config.get('min_samples_for_clustering', 20)  # Reduced from 50
            if len(embedding_matrix) < min_samples_for_clustering:
                critical_error = f"Insufficient data for clustering: {len(embedding_matrix)} < {min_samples_for_clustering}"
                results["critical_failures"].append(critical_error)
                logging.error("❌ %s", critical_error)
                return False, results
            
            # Add warning for suboptimal dataset size
            recommended_min_samples = 50
            if len(embedding_matrix) < recommended_min_samples:
                warning_msg = f"⚠️ Small dataset ({len(embedding_matrix)} samples). Results may be less stable with < {recommended_min_samples} samples"
                results["warnings"].append(warning_msg)
                logging.warning(warning_msg)
            
            logging.info("✅ Preprocessing stage successful: %d embeddings ready for training", len(embedding_matrix))
            
            # Save intermediate results if requested
            if save_intermediate_results:
                self._save_preprocessing_results(summaries, embedding_matrix, valid_indices, preprocessing_stats)
            
            # Stage 2: Clustering Training
            logging.info("STAGE 2: CLUSTERING TRAINING")
            logging.info("=" * 50)
            
            training_successful, training_results = self.clustering_trainer.run_complete_training(
                embedding_matrix=embedding_matrix,
                valid_indices=valid_indices,
                **training_config
            )
            
            results["training_results"] = training_results
            
            if not training_successful:
                critical_error = "Clustering training failed completely"
                results["critical_failures"].append(critical_error)
                logging.error("❌ %s", critical_error)
                return False, results
            
            logging.info("✅ Training stage successful")
            
            # Stage 3: Final Pipeline Statistics
            logging.info("STAGE 3: PIPELINE STATISTICS")
            logging.info("=" * 50)
            
            pipeline_duration = datetime.now() - pipeline_start_time
            
            pipeline_stats = self._compile_pipeline_statistics(
                df, preprocessing_stats, training_results, pipeline_duration
            )
            
            results["pipeline_stats"] = pipeline_stats
            results["pipeline_successful"] = True
            
            # Log final summary
            self._log_pipeline_summary(results)
            
            return True, results
            
        except Exception as e:
            critical_error = f"Pipeline failed with unexpected error: {str(e)}"
            results["critical_failures"].append(critical_error)
            logging.error("❌ Pipeline failed: %s", critical_error)
            return False, results
    
    def _compile_pipeline_statistics(self, original_df: pd.DataFrame,
                                   preprocessing_stats: Dict,
                                   training_results: Dict,
                                   pipeline_duration) -> Dict:
        """Compile comprehensive pipeline statistics"""
        
        # Extract key metrics
        original_incidents = len(original_df)
        successful_embeddings = preprocessing_stats["embedding"]["successful_embeddings"]
        clusters_found = training_results["metrics_results"]["metrics"].get("n_clusters", 0)
        noise_points = training_results["metrics_results"]["metrics"].get("n_noise_points", 0)
        
        return {
            "pipeline_overview": {
                "total_duration_minutes": pipeline_duration.total_seconds() / 60,
                "original_incidents": original_incidents,
                "incidents_clustered": successful_embeddings,
                "pipeline_success_rate": (successful_embeddings / original_incidents) * 100 if original_incidents > 0 else 0
            },
            "data_flow": {
                "input_incidents": original_incidents,
                "successful_summaries": preprocessing_stats["summarization"]["successful_summarizations"],
                "successful_embeddings": successful_embeddings,
                "incidents_in_clusters": successful_embeddings - noise_points,
                "noise_incidents": noise_points
            },
            "clustering_results": {
                "clusters_found": clusters_found,
                "cluster_quality_acceptable": training_results["quality_results"]["acceptable"],
                "silhouette_score": training_results["metrics_results"]["metrics"].get("silhouette_score", "N/A"),
                "used_fallback_params": training_results["training_results"].get("used_fallback", False)
            },
            "failure_analysis": {
                "summarization_failures": preprocessing_stats["summarization"]["failed_incidents"],
                "embedding_failures": preprocessing_stats["embedding"]["embedding_failures"],
                "training_warnings": len(training_results.get("warnings", [])),
                "critical_failures": len(training_results.get("failures", []))
            }
        }
    
    def _save_preprocessing_results(self, summaries: pd.Series, 
                                  embedding_matrix: np.ndarray,
                                  valid_indices: pd.Index,
                                  stats: Dict):
        """Save intermediate preprocessing results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save preprocessing results
            np.savez_compressed(
                f"preprocessing_results_{timestamp}.npz",
                embeddings=embedding_matrix,
                indices=valid_indices.values,
                summaries=summaries.values,
                preprocessing_stats=stats,
                timestamp=timestamp
            )
            
            logging.info("💾 Preprocessing results saved to: preprocessing_results_%s.npz", timestamp)
            
        except Exception as e:
            logging.warning("Failed to save preprocessing results: %s", str(e))
    
    def _log_pipeline_summary(self, results: Dict):
        """Log comprehensive pipeline summary"""
        logging.info("=" * 80)
        logging.info("END-TO-END PIPELINE COMPLETE - FINAL SUMMARY")
        logging.info("=" * 80)
        
        if results["pipeline_successful"]:
            stats = results["pipeline_stats"]
            
            logging.info("🎉 PIPELINE SUCCESSFUL!")
            logging.info("")
            logging.info("📊 Overall Results:")
            logging.info("  Duration: %.1f minutes", stats["pipeline_overview"]["total_duration_minutes"])
            logging.info("  Success Rate: %.1f%% (%d/%d incidents clustered)", 
                        stats["pipeline_overview"]["pipeline_success_rate"],
                        stats["data_flow"]["incidents_in_clusters"],
                        stats["pipeline_overview"]["original_incidents"])
            
            logging.info("")
            logging.info("🔄 Data Flow:")
            logging.info("  Input incidents: %d", stats["data_flow"]["input_incidents"])
            logging.info("  Successful summaries: %d", stats["data_flow"]["successful_summaries"])
            logging.info("  Successful embeddings: %d", stats["data_flow"]["successful_embeddings"])
            logging.info("  Incidents in clusters: %d", stats["data_flow"]["incidents_in_clusters"])
            logging.info("  Noise incidents: %d", stats["data_flow"]["noise_incidents"])
            
            logging.info("")
            logging.info("🎯 Clustering Results:")
            logging.info("  Clusters found: %d", stats["clustering_results"]["clusters_found"])
            logging.info("  Quality acceptable: %s", "✅" if stats["clustering_results"]["cluster_quality_acceptable"] else "⚠️")
            
            silhouette = stats["clustering_results"]["silhouette_score"]
            if silhouette != "N/A":
                logging.info("  Silhouette score: %.3f", silhouette)
            
            if stats["clustering_results"]["used_fallback_params"]:
                logging.info("  Used fallback parameters: ⚠️ Yes")
            
        else:
            logging.error("❌ PIPELINE FAILED")
            
            if results["critical_failures"]:
                logging.error("Critical Failures:")
                for failure in results["critical_failures"]:
                    logging.error("  - %s", failure)
        
        if results.get("warnings"):
            logging.warning("Warnings encountered: %d", len(results["warnings"]))
    
    def get_training_artifacts(self) -> Dict:
        """Get training artifacts for further analysis or deployment"""
        if not hasattr(self.clustering_trainer, 'trained_model') or self.clustering_trainer.trained_model is None:
            logging.warning("No trained model available")
            return {}
        
        return {
            "trained_model": self.clustering_trainer.trained_model,
            "cluster_labels": self.clustering_trainer.cluster_labels,
            "training_stats": self.clustering_trainer.training_stats,
            "preprocessing_stats": self.preprocessing_orchestrator.get_pipeline_statistics()
        }
    
    def run_training_with_parameter_search(self, df: pd.DataFrame,
                                         parameter_grid: List[Dict],
                                         preprocessing_config: Dict = None) -> Tuple[bool, Dict]:
        """
        Run training with multiple parameter sets and select the best result.
        
        Args:
            df: Raw incident DataFrame
            parameter_grid: List of parameter configurations to try
            preprocessing_config: Preprocessing configuration
            
        Returns:
            Tuple of (best_training_successful, best_results)
        """
        logging.info("Starting parameter search with %d parameter sets", len(parameter_grid))
        
        # Run preprocessing once
        preprocessing_config = preprocessing_config or {}
        summaries, embedding_matrix, valid_indices, preprocessing_stats = self.preprocessing_orchestrator.run_complete_pipeline(
            df=df, **preprocessing_config
        )
        
        if embedding_matrix is None or len(embedding_matrix) == 0:
            logging.error("Preprocessing failed - cannot run parameter search")
            return False, {"error": "Preprocessing failed"}
        
        best_score = -1
        best_results = None
        best_params = None
        
        for i, params in enumerate(parameter_grid):
            logging.info("Testing parameter set %d/%d: %s", i+1, len(parameter_grid), params)
            
            try:
                trainer = ClusteringTrainer(self.config)
                success, results = trainer.run_complete_training(
                    embedding_matrix=embedding_matrix,
                    valid_indices=valid_indices,
                    hdbscan_params=params,
                    output_dir=f"models/param_search_{i+1}"
                )
                
                if success:
                    # Use silhouette score as primary metric, fallback to cluster count
                    score = results["metrics_results"]["metrics"].get("silhouette_score", 0)
                    if score == 0:  # No silhouette score available
                        score = results["metrics_results"]["metrics"].get("n_clusters", 0) / 100  # Normalize
                    
                    logging.info("Parameter set %d score: %.3f", i+1, score)
                    
                    if score > best_score:
                        best_score = score
                        best_results = results
                        best_params = params
                        logging.info("✅ New best parameter set found!")
                
            except Exception as e:
                logging.warning("Parameter set %d failed: %s", i+1, str(e))
        
        if best_results is not None:
            logging.info("Parameter search complete. Best parameters: %s (score: %.3f)", best_params, best_score)
            return True, {
                "best_results": best_results,
                "best_params": best_params,
                "best_score": best_score,
                "preprocessing_stats": preprocessing_stats
            }
        else:
            logging.error("Parameter search failed - no successful parameter sets")
            return False, {"error": "All parameter sets failed"}