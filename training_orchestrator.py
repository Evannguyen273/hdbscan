#!/usr/bin/env python3
"""
Training Orchestrator - Coordinates cumulative training workflow
Manages the complete pipeline from data loading to versioned table creation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Local imports
from training_pipeline import EnhancedTrainingPipeline
from clustering_trainer import ClusteringTrainer
from config.config import get_config
from logging_setup import setup_detailed_logging


class TrainingOrchestrator:
    """
    Orchestrates the complete cumulative training workflow
    Coordinates data loading, training, and versioned table creation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the training orchestrator"""
        setup_detailed_logging(logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.config = get_config(config_path)
        
        # Initialize components
        self.training_pipeline = EnhancedTrainingPipeline(config_path)
        self.clustering_trainer = ClusteringTrainer(config_path)
        
        self.logger.info("Training Orchestrator initialized for cumulative approach")
    
    def run_cumulative_training_cycle(self, tech_centers: List[str], 
                                    year: int, quarter: str) -> Dict[str, Any]:
        """
        Run complete cumulative training cycle for multiple tech centers
        
        Args:
            tech_centers: List of tech center names
            year: Training year
            quarter: Training quarter (q1, q2, q3, q4)
            
        Returns:
            Dictionary with training results for all tech centers
        """
        try:
            self.logger.info(f"Starting cumulative training cycle: {year} Q{quarter}")
            self.logger.info(f"Tech centers: {len(tech_centers)}")
            self.logger.info(f"Training approach: 24-month rolling window")
            
            cycle_results = {
                "cycle_id": f"{year}_{quarter}",
                "training_approach": "cumulative_24_months",
                "start_timestamp": datetime.now().isoformat(),
                "tech_centers": {},
                "summary": {
                    "total_centers": len(tech_centers),
                    "successful_centers": 0,
                    "failed_centers": 0,
                    "total_domains_created": 0,
                    "total_records_processed": 0,
                    "versioned_tables_created": []
                }
            }
            
            for i, tech_center in enumerate(tech_centers):
                try:
                    self.logger.info(f"Processing tech center {i+1}/{len(tech_centers)}: {tech_center}")
                    
                    # Run training for individual tech center
                    center_result = self.run_tech_center_training(tech_center, year, quarter)
                    
                    # Update cycle results
                    cycle_results["tech_centers"][tech_center] = center_result
                    cycle_results["summary"]["successful_centers"] += 1
                    cycle_results["summary"]["total_domains_created"] += center_result.get("domains_count", 0)
                    cycle_results["summary"]["total_records_processed"] += center_result.get("records_processed", 0)
                    
                    if "table_name" in center_result:
                        cycle_results["summary"]["versioned_tables_created"].append(center_result["table_name"])
                    
                    self.logger.info(f"✅ {tech_center}: {center_result.get('domains_count', 0)} domains, {center_result.get('records_processed', 0)} records")
                    
                except Exception as e:
                    self.logger.error(f"❌ Training failed for {tech_center}: {e}")
                    cycle_results["tech_centers"][tech_center] = {
                        "status": "failed",
                        "error": str(e),
                        "domains_count": 0,
                        "records_processed": 0
                    }
                    cycle_results["summary"]["failed_centers"] += 1
            
            cycle_results["end_timestamp"] = datetime.now().isoformat()
            cycle_results["duration_minutes"] = self._calculate_duration(
                cycle_results["start_timestamp"], cycle_results["end_timestamp"]
            )
            
            # Save cycle summary
            self._save_cycle_summary(cycle_results, year, quarter)
            
            self.logger.info(f"Cumulative training cycle completed: {cycle_results['summary']['successful_centers']}/{cycle_results['summary']['total_centers']} successful")
            return cycle_results
            
        except Exception as e:
            self.logger.error(f"Training cycle failed: {e}")
            raise
    
    def run_tech_center_training(self, tech_center: str, year: int, quarter: str) -> Dict[str, Any]:
        """
        Run complete training workflow for a single tech center
        
        Steps:
        1. Load 24-month cumulative data
        2. Train HDBSCAN clustering
        3. Generate domain groupings
        4. Create versioned BigQuery table
        5. Save model artifacts
        """
        try:
            self.logger.info(f"Starting training workflow for {tech_center}")
            
            # 1. Load cumulative training data (24-month window)
            training_data = self._load_cumulative_training_data(tech_center, year, quarter)
            
            if len(training_data["embeddings"]) < 100:  # Minimum data requirement
                raise ValueError(f"Insufficient data for training: {len(training_data['embeddings'])} incidents")
            
            # 2. Train clustering models
            clustering_results = self.clustering_trainer.train_hdbscan_model(
                embeddings=training_data["embeddings"],
                tech_center=tech_center,
                year=year,
                quarter=quarter
            )
            
            # 3. Run enhanced pipeline for domain grouping and table creation
            domain_results = self.training_pipeline.run_training_with_domains(
                tech_center=tech_center,
                quarter=quarter,
                year=year
            )
            
            # 4. Combine results and create comprehensive summary
            training_result = {
                "status": "success",
                "tech_center": tech_center,
                "training_year": year,
                "training_quarter": quarter,
                "training_window": self._get_training_window(year, quarter),
                "records_processed": len(training_data["embeddings"]),
                "clusters_found": clustering_results["n_clusters"],
                "domains_count": domain_results["summary"]["domains_count"],
                "table_name": domain_results["summary"]["bigquery_table"]["table_name"],
                "model_version": domain_results["summary"]["bigquery_table"]["model_version"],
                "training_metrics": clustering_results["training_summary"]["evaluation_metrics"],
                "model_paths": clustering_results["training_summary"]["model_paths"],
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Training completed for {tech_center}")
            return training_result
            
        except Exception as e:
            self.logger.error(f"Training failed for {tech_center}: {e}")
            return {
                "status": "failed",
                "tech_center": tech_center,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _load_cumulative_training_data(self, tech_center: str, year: int, quarter: str) -> Dict[str, Any]:
        """
        Load 24-month cumulative training data from preprocessed_incidents table
        """
        try:
            # Calculate 24-month window
            training_window = self._get_training_window(year, quarter)
            
            self.logger.info(f"Loading cumulative data for {tech_center}")
            self.logger.info(f"Training window: {training_window['start_date']} to {training_window['end_date']}")
            
            # Mock data loading (in real implementation, query BigQuery)
            # Query would be:
            # SELECT number, sys_created_on, combined_incidents_summary, embedding
            # FROM preprocessed_incidents 
            # WHERE tech_center = ? AND sys_created_on BETWEEN ? AND ?
            # ORDER BY sys_created_on
            
            # Calculate expected data size (mock)
            days_in_window = (datetime.strptime(training_window['end_date'], '%Y-%m-%d') - 
                            datetime.strptime(training_window['start_date'], '%Y-%m-%d')).days
            expected_incidents = int(days_in_window * 120)  # ~120 incidents per day (mock)
            
            # Generate mock embeddings for demonstration
            mock_embeddings = np.random.rand(expected_incidents, 1536)
            
            training_data = {
                "embeddings": mock_embeddings,
                "incident_count": expected_incidents,
                "tech_center": tech_center,
                "training_window": training_window,
                "data_source": "preprocessed_incidents"
            }
            
            self.logger.info(f"Loaded {expected_incidents} incidents for cumulative training")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Failed to load training data for {tech_center}: {e}")
            raise
    
    def _get_training_window(self, year: int, quarter: str) -> Dict[str, str]:
        """Calculate 24-month training window dates"""
        # Start date: 24 months before the end of current quarter
        start_year = year - 2  # Go back 24 months
        
        quarter_start_map = {"q1": "01-01", "q2": "04-01", "q3": "07-01", "q4": "10-01"}
        quarter_end_map = {"q1": "03-31", "q2": "06-30", "q3": "09-30", "q4": "12-31"}
        
        start_date = f"{start_year}-{quarter_start_map.get(quarter, '01-01')}"
        end_date = f"{year}-{quarter_end_map.get(quarter, '12-31')}"
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "duration_months": 24,
            "approach": "cumulative"
        }
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration in minutes"""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return (end - start).total_seconds() / 60
        except:
            return 0.0
    
    def _save_cycle_summary(self, cycle_results: Dict[str, Any], year: int, quarter: str):
        """Save training cycle summary"""
        try:
            # Create cycle results directory
            results_dir = Path("results") / "training_cycles"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cycle summary
            summary_file = results_dir / f"cycle_{year}_{quarter}_summary.json"
            import json
            with open(summary_file, 'w') as f:
                json.dump(cycle_results, f, indent=2, default=str)
            
            self.logger.info(f"Cycle summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save cycle summary: {e}")


class ScheduledTrainingRunner:
    """Runs scheduled training cycles"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.orchestrator = TrainingOrchestrator(config_path)
        self.logger = logging.getLogger(__name__)
    
    def run_quarterly_training(self, year: int, quarter: str, 
                             tech_centers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run quarterly training for all or specified tech centers
        
        This is the main entry point for scheduled training runs
        """
        if tech_centers is None:
            tech_centers = [
                "BT-TC-Data Analytics",
                "BT-TC-Network Operations", 
                "BT-TC-Product Development & Engineering",
                "BT-TC-DevOps Engineering",
                "BT-TC-Business Intelligence"
            ]
        
        self.logger.info(f"Starting quarterly training: {year} Q{quarter}")
        self.logger.info(f"Tech centers: {len(tech_centers)}")
        
        # Run cumulative training cycle
        results = self.orchestrator.run_cumulative_training_cycle(
            tech_centers=tech_centers,
            year=year,
            quarter=quarter
        )
        
        # Print summary
        self._print_training_summary(results)
        
        return results
    
    def _print_training_summary(self, results: Dict[str, Any]):
        """Print formatted training summary"""
        print(f"\n{'='*60}")
        print(f"QUARTERLY TRAINING SUMMARY - {results['cycle_id']}")
        print(f"{'='*60}")
        print(f"Training Approach: {results['training_approach']}")
        print(f"Duration: {results.get('duration_minutes', 0):.1f} minutes")
        print(f"Tech Centers: {results['summary']['successful_centers']}/{results['summary']['total_centers']} successful")
        print(f"Total Domains: {results['summary']['total_domains_created']}")
        print(f"Total Records: {results['summary']['total_records_processed']:,}")
        print()
        print("Versioned Tables Created:")
        for table in results['summary']['versioned_tables_created']:
            print(f"  • {table}")
        print(f"{'='*60}")


def main():
    """Example usage"""
    runner = ScheduledTrainingRunner()
    
    # Run quarterly training
    results = runner.run_quarterly_training(
        year=2024,
        quarter="q4",
        tech_centers=["BT-TC-Data Analytics", "BT-TC-Network Operations"]
    )
    
    print(f"Training completed: {results['summary']['successful_centers']} centers successful")


if __name__ == "__main__":
    main()