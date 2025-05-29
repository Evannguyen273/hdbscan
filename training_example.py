# filepath: c:\Users\Bachn\OneDrive\Desktop\my_projects\hdbscan\training_example.py
"""
Complete training pipeline example with comprehensive failure handling.
Shows how to train clustering models from raw incident data with guardrails.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict

# Setup logging
from logging_setup import setup_detailed_logging
setup_detailed_logging(logging.INFO)

# Import training components
from training.training_orchestrator import TrainingOrchestrator
from config import Config  # Assume you have a config module

def run_basic_training_example():
    """Example of running the complete training pipeline"""
    
    print("="*80)
    print("COMPLETE TRAINING PIPELINE EXAMPLE")
    print("="*80)
    
    # Create sample incident data
    sample_data = create_sample_incident_data(size=200)  # Larger dataset for clustering
    df = pd.DataFrame(sample_data)
    
    print(f"Input data: {len(df)} incidents")
    
    # Initialize the training orchestrator
    config = Config()  # Your configuration
    orchestrator = TrainingOrchestrator(config)
    
    # Configure the pipeline
    preprocessing_config = {
        "summarization_batch_size": 25,
        "embedding_batch_size": 50,
        "use_batch_embedding_api": True
    }
      training_config = {
        "hdbscan_params": {
            "min_cluster_size": 8,  # Reduced for smaller datasets
            "min_samples": 3,       # Reduced for smaller datasets
            "cluster_selection_epsilon": 0.0,
            "metric": "euclidean"
        },
        "fallback_params_list": [
            {"min_cluster_size": 6, "min_samples": 2, "metric": "euclidean"},
            {"min_cluster_size": 4, "min_samples": 1, "metric": "cosine"},
            {"min_cluster_size": 3, "min_samples": 1, "metric": "euclidean"}
        ],
        "preprocessing_config": {
            "apply_scaling": True, 
            "apply_pca": False
        },
        "output_dir": "models/basic_training"
    }
    
    # Run the complete training pipeline
    print("\nStarting complete training pipeline...")
    
    try:
        success, results = orchestrator.run_end_to_end_training(
            df=df,
            preprocessing_config=preprocessing_config,
            training_config=training_config,
            save_intermediate_results=True
        )
        
        if success:
            print("\n" + "="*80)
            print("TRAINING SUCCESSFUL!")
            print("="*80)
            
            stats = results["pipeline_stats"]
            
            print(f"📊 Pipeline Overview:")
            print(f"  - Duration: {stats['pipeline_overview']['total_duration_minutes']:.1f} minutes")
            print(f"  - Success Rate: {stats['pipeline_overview']['pipeline_success_rate']:.1f}%")
            print(f"  - Incidents Clustered: {stats['data_flow']['incidents_in_clusters']}/{stats['pipeline_overview']['original_incidents']}")
            
            print(f"\n🎯 Clustering Results:")
            print(f"  - Clusters Found: {stats['clustering_results']['clusters_found']}")
            print(f"  - Noise Points: {stats['data_flow']['noise_incidents']}")
            print(f"  - Quality Acceptable: {'✅' if stats['clustering_results']['cluster_quality_acceptable'] else '⚠️'}")
            
            if stats['clustering_results']['silhouette_score'] != "N/A":
                print(f"  - Silhouette Score: {stats['clustering_results']['silhouette_score']:.3f}")
            
            # Show failure breakdown if any
            failure_analysis = stats['failure_analysis']
            total_failures = (failure_analysis['summarization_failures'] + 
                             failure_analysis['embedding_failures'])
            
            if total_failures > 0:
                print(f"\n⚠️ Failure Analysis:")
                print(f"  - Summarization Failures: {failure_analysis['summarization_failures']}")
                print(f"  - Embedding Failures: {failure_analysis['embedding_failures']}")
                print(f"  - Training Warnings: {failure_analysis['training_warnings']}")
            
            # Get training artifacts
            artifacts = orchestrator.get_training_artifacts()
            if artifacts:
                print(f"\n🔧 Training Artifacts Available:")
                print(f"  - Trained Model: ✅")
                print(f"  - Cluster Labels: ✅ ({len(artifacts['cluster_labels'])} labels)")
                print(f"  - Training Stats: ✅")
            
        else:
            print("\n" + "="*80)
            print("TRAINING FAILED!")
            print("="*80)
            
            print(f"❌ Critical Failures:")
            for failure in results["critical_failures"]:
                print(f"  - {failure}")
            
            if results.get("warnings"):
                print(f"\n⚠️ Warnings:")
                for warning in results["warnings"]:
                    print(f"  - {warning}")
        
        return success, results
        
    except Exception as e:
        logging.error("Training pipeline failed: %s", e)
        print(f"\n❌ Training pipeline failed: {e}")
        return False, None

def demonstrate_parameter_search():
    """Demonstrate parameter search for optimal clustering"""
    
    print("\n" + "="*80)
    print("PARAMETER SEARCH EXAMPLE")
    print("="*80)
    
    # Create sample data
    sample_data = create_sample_incident_data(size=150)
    df = pd.DataFrame(sample_data)
    
    # Define parameter grid to search
    parameter_grid = [
        {"min_cluster_size": 15, "min_samples": 5, "metric": "euclidean"},
        {"min_cluster_size": 10, "min_samples": 3, "metric": "euclidean"},
        {"min_cluster_size": 12, "min_samples": 4, "metric": "cosine"},
        {"min_cluster_size": 8, "min_samples": 2, "metric": "euclidean"},
    ]
    
    config = Config()
    orchestrator = TrainingOrchestrator(config)
    
    print(f"Testing {len(parameter_grid)} parameter combinations...")
    
    try:
        success, results = orchestrator.run_training_with_parameter_search(
            df=df,
            parameter_grid=parameter_grid,
            preprocessing_config={"summarization_batch_size": 20}
        )
        
        if success:
            best_params = results["best_params"]
            best_score = results["best_score"]
            
            print(f"\n✅ Parameter search successful!")
            print(f"Best parameters: {best_params}")
            print(f"Best score: {best_score:.3f}")
            
            # Show results from best parameters
            best_results = results["best_results"]
            metrics = best_results["metrics_results"]["metrics"]
            
            print(f"\nBest Results:")
            print(f"  - Clusters: {metrics.get('n_clusters', 0)}")
            print(f"  - Noise points: {metrics.get('n_noise_points', 0)}")
            print(f"  - Silhouette score: {metrics.get('silhouette_score', 'N/A')}")
            
        else:
            print(f"\n❌ Parameter search failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n❌ Parameter search failed: {e}")

def demonstrate_training_failure_scenarios():
    """Demonstrate how the system handles various training failure scenarios"""
    
    print("\n" + "="*80)
    print("TRAINING FAILURE SCENARIO DEMONSTRATIONS")
    print("="*80)
    
    config = Config()
    orchestrator = TrainingOrchestrator(config)
    
    # Scenario 1: Insufficient data
    print("\n1. Testing with insufficient data...")
    small_data = create_sample_incident_data(size=10)  # Too small for clustering
    df_small = pd.DataFrame(small_data)
    
    try:
        success, results = orchestrator.run_end_to_end_training(
            df=df_small,
            preprocessing_config={"summarization_batch_size": 5},
            training_config={
                "hdbscan_params": {"min_cluster_size": 15, "min_samples": 5},
                "fallback_params_list": []
            }
        )
        
        print(f"   Result: {'Success' if success else 'Failed as expected'}")
        if not success and results["critical_failures"]:
            print(f"   Reason: {results['critical_failures'][0]}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Scenario 2: Bad parameters
    print("\n2. Testing with invalid parameters...")
    normal_data = create_sample_incident_data(size=100)
    df_normal = pd.DataFrame(normal_data)
    
    try:
        success, results = orchestrator.run_end_to_end_training(
            df=df_normal,
            training_config={
                "hdbscan_params": {"min_cluster_size": -5, "min_samples": 0},  # Invalid params
                "fallback_params_list": [
                    {"min_cluster_size": 8, "min_samples": 2}  # Valid fallback
                ]
            }
        )
        
        print(f"   Result: {'Success with fallback' if success else 'Failed'}")
        if success:
            used_fallback = results["training_results"]["training_results"].get("used_fallback", False)
            print(f"   Used fallback parameters: {used_fallback}")
        
    except Exception as e:
        print(f"   Error: {e}")

def demonstrate_adaptive_parameter_selection():
    """Demonstrate automatic parameter selection based on dataset size"""
    
    print("\n" + "="*80)
    print("ADAPTIVE PARAMETER SELECTION EXAMPLE")
    print("="*80)
    
    from training.clustering_trainer import ClusteringTrainer
    
    # Test different dataset sizes
    test_sizes = [25, 45, 75, 150, 300]
    
    for size in test_sizes:
        print(f"\nDataset size: {size} samples")
        
        suggested = ClusteringTrainer.suggest_parameters_for_dataset_size(size)
        
        print(f"  Category: {suggested['dataset_size_category']}")
        print(f"  Primary params: {suggested['primary_params']}")
        print(f"  Fallback options: {len(suggested['fallback_params'])} sets")
        
        # Show first fallback as example
        if suggested['fallback_params']:
            print(f"  First fallback: {suggested['fallback_params'][0]}")

def run_adaptive_training_example():
    """Example using adaptive parameter selection"""
    
    print("\n" + "="*80)
    print("ADAPTIVE TRAINING EXAMPLE")
    print("="*80)
    
    # Create a small dataset to test adaptive parameters
    sample_data = create_sample_incident_data(size=35)  # Small dataset
    df = pd.DataFrame(sample_data)
    
    print(f"Input data: {len(df)} incidents (small dataset)")
    
    # Get suggested parameters for this dataset size
    from training.clustering_trainer import ClusteringTrainer
    suggested = ClusteringTrainer.suggest_parameters_for_dataset_size(len(df))
    
    print(f"Suggested parameters for {len(df)} samples:")
    print(f"  Category: {suggested['dataset_size_category']}")
    print(f"  Primary: {suggested['primary_params']}")
    
    # Initialize the training orchestrator
    config = Config()
    orchestrator = TrainingOrchestrator(config)
    
    # Use suggested parameters
    training_config = {
        "hdbscan_params": suggested['primary_params'],
        "fallback_params_list": suggested['fallback_params'],
        "preprocessing_config": {
            "apply_scaling": True, 
            "apply_pca": False
        },
        "output_dir": "models/adaptive_training"
    }
    
    print("\nRunning training with adaptive parameters...")
    
    try:
        success, results = orchestrator.run_end_to_end_training(
            df=df,
            preprocessing_config={"summarization_batch_size": 15},
            training_config=training_config,
            save_intermediate_results=True
        )
        
        if success:
            print("\n✅ Adaptive training successful!")
            
            stats = results["pipeline_stats"]
            training_results = results["training_results"]["training_results"]
            
            print(f"📊 Results:")
            print(f"  - Clusters Found: {stats['clustering_results']['clusters_found']}")
            print(f"  - Used Fallback: {'Yes' if training_results.get('used_fallback', False) else 'No'}")
            
            if training_results.get('used_fallback', False):
                used_params = training_results.get('successful_params', {})
                print(f"  - Successful Parameters: {used_params}")
            
            print(f"  - Quality Acceptable: {'✅' if stats['clustering_results']['cluster_quality_acceptable'] else '⚠️'}")
            
        else:
            print("\n❌ Adaptive training failed")
            for failure in results["critical_failures"]:
                print(f"  - {failure}")
                
        return success, results
        
    except Exception as e:
        print(f"\n❌ Adaptive training failed: {e}")
        return False, None

def create_sample_incident_data(size: int = 100) -> Dict:
    """Create realistic sample incident data for testing"""
    
    # Define incident patterns for different types
    patterns = [
        # SharePoint issues
        {
            "short_desc_templates": [
                "SharePoint site not loading",
                "SharePoint access denied",
                "SharePoint sync issues",
                "SharePoint document upload failed"
            ],
            "desc_templates": [
                "Users cannot access SharePoint site. Getting timeout errors when trying to load pages.",
                "SharePoint authentication failing. Users see access denied message.",
                "SharePoint sync client not working. Files not syncing to local drive.",
                "Document upload to SharePoint library fails with error message."
            ],
            "business_service": "SharePoint Online"
        },
        # Email issues
        {
            "short_desc_templates": [
                "Outlook not receiving emails",
                "Email delivery delayed",
                "Outlook crashes on startup",
                "Email attachment issues"
            ],
            "desc_templates": [
                "Outlook client not receiving new emails. Inbox shows no new messages.",
                "Email delivery experiencing significant delays. Messages arriving hours late.",
                "Outlook application crashes immediately after startup. Cannot access email.",
                "Cannot open or download email attachments. Getting file corruption errors."
            ],
            "business_service": "Exchange Online"
        },
        # Database issues
        {
            "short_desc_templates": [
                "Database connection timeout",
                "SQL query performance slow",
                "Database backup failed",
                "Database server unresponsive"
            ],
            "desc_templates": [
                "Application cannot connect to database. Connection timeout after 30 seconds.",
                "SQL queries running extremely slow. Performance degraded significantly.",
                "Automated database backup job failed last night. Backup file not created.",
                "Database server not responding to requests. All database operations failing."
            ],
            "business_service": "SQL Server"
        }
    ]
    
    incidents = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(size):
        # Choose random pattern
        pattern = np.random.choice(patterns)
        
        # Add some variation for clustering
        variation_factor = np.random.choice([0, 1, 2])  # 0=exact, 1=slight variation, 2=more variation
        
        short_desc = np.random.choice(pattern["short_desc_templates"])
        desc = np.random.choice(pattern["desc_templates"])
        
        # Add variations to create subclusters
        if variation_factor == 1:
            desc += " Additional context: user reports this happens frequently."
        elif variation_factor == 2:
            desc += " Error occurs specifically during peak hours. Multiple users affected."
        
        incidents.append({
            "number": f"INC{i+1000:07d}",
            "short_description": short_desc,
            "description": desc,
            "business_service": pattern["business_service"]
        })
    
    return {
        "number": [inc["number"] for inc in incidents],
        "short_description": [inc["short_description"] for inc in incidents],
        "description": [inc["description"] for inc in incidents],
        "business_service": [inc["business_service"] for inc in incidents]
    }

def show_expected_training_logs():
    """Show what the training logs look like"""
    
    print("\n" + "="*80)
    print("EXAMPLE TRAINING LOG OUTPUT")
    print("="*80)
    
    example_logs = """
2024-01-15 15:00:00 | INFO     | ================================================================================
2024-01-15 15:00:00 | INFO     | STARTING END-TO-END TRAINING PIPELINE
2024-01-15 15:00:00 | INFO     | ================================================================================
2024-01-15 15:00:00 | INFO     | Input: 200 raw incidents
2024-01-15 15:00:00 | INFO     | STAGE 1: PREPROCESSING PIPELINE
2024-01-15 15:00:00 | INFO     | ==================================================
2024-01-15 15:00:00 | INFO     | Starting complete preprocessing pipeline for 200 incidents
2024-01-15 15:02:30 | INFO     | ✅ Preprocessing stage successful: 185 embeddings ready for training
2024-01-15 15:02:30 | INFO     | STAGE 2: CLUSTERING TRAINING
2024-01-15 15:02:30 | INFO     | ==================================================
2024-01-15 15:02:30 | INFO     | ============================================================
2024-01-15 15:02:30 | INFO     | STARTING CLUSTERING TRAINING PIPELINE
2024-01-15 15:02:30 | INFO     | ============================================================
2024-01-15 15:02:31 | INFO     | Stage 1: Validating training data...
2024-01-15 15:02:31 | INFO     | ✓ Data validation passed
2024-01-15 15:02:31 | INFO     | Stage 2: Preprocessing embeddings...
2024-01-15 15:02:31 | INFO     | Applying standardization to embeddings...
2024-01-15 15:02:31 | INFO     | ✓ Standardization applied successfully
2024-01-15 15:02:31 | INFO     | Stage 3: Training HDBSCAN model...
2024-01-15 15:02:31 | INFO     | Training HDBSCAN with primary parameters...
2024-01-15 15:02:32 | INFO     | ✓ Primary parameters successful: 8 clusters, 23 noise points
2024-01-15 15:02:32 | INFO     | Stage 4: Calculating clustering metrics...
2024-01-15 15:02:32 | INFO     | ✓ Silhouette score: 0.347
2024-01-15 15:02:32 | INFO     | ✓ Calinski-Harabasz score: 89.234
2024-01-15 15:02:32 | INFO     | Stage 5: Validating clustering quality...
2024-01-15 15:02:32 | INFO     | Stage 6: Saving training results...
2024-01-15 15:02:33 | INFO     | ✓ Model saved to: models/hdbscan_model_20240115_150233.joblib
2024-01-15 15:02:33 | INFO     | ✅ Training stage successful
2024-01-15 15:02:33 | INFO     | ================================================================================
2024-01-15 15:02:33 | INFO     | END-TO-END PIPELINE COMPLETE - FINAL SUMMARY
2024-01-15 15:02:33 | INFO     | ================================================================================
2024-01-15 15:02:33 | INFO     | 🎉 PIPELINE SUCCESSFUL!
2024-01-15 15:02:33 | INFO     | 📊 Overall Results:
2024-01-15 15:02:33 | INFO     |   Duration: 2.6 minutes
2024-01-15 15:02:33 | INFO     |   Success Rate: 81.0% (162/200 incidents clustered)
2024-01-15 15:02:33 | INFO     | 🎯 Clustering Results:
2024-01-15 15:02:33 | INFO     |   Clusters found: 8
2024-01-15 15:02:33 | INFO     |   Quality acceptable: ✅
2024-01-15 15:02:33 | INFO     |   Silhouette score: 0.347
    """
    
    print(example_logs)

if __name__ == "__main__":
    # Show expected log output
    show_expected_training_logs()
    
    # Run basic training example
    success, results = run_basic_training_example()
    
    # Demonstrate parameter search
    if success:  # Only if basic training worked
        demonstrate_parameter_search()
    
    # Demonstrate failure scenarios
    demonstrate_training_failure_scenarios()
    
    # Demonstrate adaptive parameter selection
    demonstrate_adaptive_parameter_selection()
    
    # Run adaptive training example
    run_adaptive_training_example()
    
    print(f"\n🎉 Training examples complete! Check the log files and models/ directory for results.")