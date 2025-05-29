"""
Comprehensive examples showing what happens when each validation stage fails.
This demonstrates the exact behavior and error handling for all guardrails.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List

# Setup logging to see detailed failure messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_failing_datasets():
    """Create various datasets that will fail different validation stages"""
    
    return {
        # 🔍 DATA VALIDATION FAILURES
        "empty_dataset": pd.DataFrame(),  # No data at all
        
        "insufficient_samples": pd.DataFrame({
            'incident_description': ['Error 1'] * 15,  # Only 15 samples (< 20 minimum)
            'embeddings': [np.random.rand(384).tolist()] * 15
        }),
        
        "nan_embeddings": pd.DataFrame({
            'incident_description': ['Error ' + str(i) for i in range(50)],
            'embeddings': [[np.nan] * 384 if i % 5 == 0 else np.random.rand(384).tolist() for i in range(50)]
        }),
        
        "infinite_embeddings": pd.DataFrame({
            'incident_description': ['Error ' + str(i) for i in range(50)],
            'embeddings': [[np.inf] * 384 if i % 5 == 0 else np.random.rand(384).tolist() for i in range(50)]
        }),
          "low_variance_embeddings": pd.DataFrame({
            'incident_description': ['Same error repeated'] * 50,  # Realistic: same issue reported multiple times
            'embeddings': [[0.5] * 384] * 50  # All identical embeddings - this is actually normal for repetitive issues
        }),
        
        "too_many_duplicates": pd.DataFrame({
            'incident_description': ['Error ' + str(i) for i in range(50)],
            'embeddings': [np.random.rand(384).tolist() if i < 10 else [0.5] * 384 for i in range(50)]  # 80% duplicates
        }),
        
        # ⚙️ PARAMETER VALIDATION FAILURES
        "invalid_parameters": {
            'data': pd.DataFrame({
                'incident_description': ['Error ' + str(i) for i in range(50)],
                'embeddings': [np.random.rand(384).tolist() for i in range(50)]
            }),
            'bad_params': {
                'min_cluster_size': -5,  # Invalid: negative
                'min_samples': 'invalid',  # Invalid: not integer
                'metric': 'invalid_metric'  # Invalid: unsupported metric
            }
        },
        
        # 🎯 TRAINING FAILURES
        "extreme_parameters": {
            'data': pd.DataFrame({
                'incident_description': ['Error ' + str(i) for i in range(30)],
                'embeddings': [np.random.rand(384).tolist() for i in range(30)]
            }),
            'bad_params': {
                'min_cluster_size': 25,  # Too large for 30 samples
                'min_samples': 30,      # Larger than min_cluster_size
                'metric': 'euclidean'
            }
        },
        
        # 📊 QUALITY VALIDATION FAILURES
        "poor_quality_result": pd.DataFrame({
            'incident_description': ['Error ' + str(i) for i in range(100)],
            'embeddings': [np.random.rand(2).tolist() for i in range(100)]  # Very low dimensions, poor separation
        })
    }

def demonstrate_data_validation_failures():
    """Show what happens when data validation fails"""
    
    print("\n" + "="*80)
    print("🔍 DATA VALIDATION FAILURE EXAMPLES")
    print("="*80)
    
    from training.clustering_trainer import ClusteringTrainer
    
    # Mock config
    config = {
        'min_samples_for_training': 20,
        'min_embedding_dimensions': 50,
        'max_memory_usage_gb': 8.0
    }
    
    trainer = ClusteringTrainer(config)
    failing_datasets = create_failing_datasets()
    
    # Test each failing scenario
    scenarios = [
        ("Empty Dataset", "empty_dataset"),
        ("Insufficient Samples", "insufficient_samples"), 
        ("NaN Embeddings", "nan_embeddings"),
        ("Infinite Embeddings", "infinite_embeddings"),
        ("Low Variance Embeddings", "low_variance_embeddings"),
        ("Too Many Duplicates", "too_many_duplicates")
    ]
    
    for scenario_name, dataset_key in scenarios:
        print(f"\n--- Testing: {scenario_name} ---")
        
        try:
            df = failing_datasets[dataset_key]
            
            if len(df) == 0:
                print("❌ RESULT: Training immediately fails - no data to process")
                continue
                
            # Convert embeddings to matrix
            embedding_matrix = np.array(df['embeddings'].tolist())
            valid_indices = df.index
            
            # Test validation
            is_valid, validation_errors = trainer.validate_training_data(embedding_matrix, valid_indices)
            
            if not is_valid:
                print(f"❌ RESULT: Data validation failed")
                print(f"   ERRORS: {validation_errors}")
                print(f"   CONSEQUENCE: Training pipeline stops here")
                print(f"   USER SEES: 'Training failed during data validation stage'")
            else:
                print(f"⚠️ RESULT: Validation passed with warnings")
                print(f"   WARNINGS: {trainer.training_warnings}")
                
        except Exception as e:
            print(f"💥 RESULT: Critical error during validation")
            print(f"   ERROR: {str(e)}")
            print(f"   CONSEQUENCE: Complete pipeline failure")

def demonstrate_parameter_validation_failures():
    """Show what happens when parameter validation fails"""
    
    print("\n" + "="*80)
    print("⚙️ PARAMETER VALIDATION FAILURE EXAMPLES")
    print("="*80)
    
    from training.clustering_trainer import ClusteringTrainer
    
    config = {'min_samples_for_training': 20, 'min_embedding_dimensions': 50}
    trainer = ClusteringTrainer(config)
    failing_datasets = create_failing_datasets()
    
    # Test invalid parameters
    print("\n--- Testing: Invalid Parameters ---")
    
    scenario = failing_datasets["invalid_parameters"]
    df = scenario['data']
    bad_params = scenario['bad_params']
    
    embedding_matrix = np.array(df['embeddings'].tolist())
    
    try:
        param_valid, param_errors = trainer.validate_hdbscan_parameters(embedding_matrix, **bad_params)
        
        if not param_valid:
            print(f"❌ RESULT: Parameter validation failed")
            print(f"   ERRORS: {param_errors}")
            print(f"   CONSEQUENCE: Training stops before HDBSCAN is even attempted")
            print(f"   USER SEES: 'Invalid parameters provided' + specific error details")
        
    except Exception as e:
        print(f"💥 RESULT: Parameter validation crashed")
        print(f"   ERROR: {str(e)}")

def demonstrate_training_failures():
    """Show what happens when training itself fails"""
    
    print("\n" + "="*80)
    print("🎯 TRAINING STAGE FAILURE EXAMPLES") 
    print("="*80)
    
    from training.clustering_trainer import ClusteringTrainer
    
    config = {'min_samples_for_training': 20, 'min_embedding_dimensions': 2}
    trainer = ClusteringTrainer(config)
    failing_datasets = create_failing_datasets()
    
    print("\n--- Testing: Extreme Parameters (Primary + Fallbacks Fail) ---")
    
    scenario = failing_datasets["extreme_parameters"]
    df = scenario['data']
    bad_params = scenario['bad_params']
    
    embedding_matrix = np.array(df['embeddings'].tolist())
    
    # Primary parameters will fail
    fallback_params = [
        {"min_cluster_size": 20, "min_samples": 25},  # Still too large
        {"min_cluster_size": 15, "min_samples": 20},  # Still too large
        {"min_cluster_size": 10, "min_samples": 15}   # Still problematic
    ]
    
    try:
        trained_model, training_info = trainer.train_hdbscan_with_fallbacks(
            embedding_matrix, bad_params, fallback_params
        )
        
        if trained_model is None:
            print(f"❌ RESULT: All training attempts failed")
            print(f"   PRIMARY FAILED: {training_info['training_errors'][0]}")
            print(f"   FALLBACKS TRIED: {len(fallback_params)}")
            print(f"   ALL ERRORS: {training_info['training_errors']}")
            print(f"   CONSEQUENCE: No model produced, training pipeline fails")
            print(f"   USER SEES: 'Training failed - all parameter sets unsuccessful'")
        
    except Exception as e:
        print(f"💥 RESULT: Training crashed completely")
        print(f"   ERROR: {str(e)}")

def demonstrate_quality_validation_failures():
    """Show what happens when quality validation fails"""
    
    print("\n" + "="*80)
    print("📊 QUALITY VALIDATION FAILURE EXAMPLES")
    print("="*80)
    
    from training.clustering_trainer import ClusteringTrainer
    
    config = {'min_samples_for_training': 20, 'min_embedding_dimensions': 2}
    trainer = ClusteringTrainer(config)
    failing_datasets = create_failing_datasets()
    
    print("\n--- Testing: Poor Quality Results ---")
    
    df = failing_datasets["poor_quality_result"]
    embedding_matrix = np.array(df['embeddings'].tolist())
    
    # Use parameters that will produce poor clustering
    params = {"min_cluster_size": 5, "min_samples": 2, "metric": "euclidean"}
    
    try:
        # Train model (will likely produce poor results due to low-dimensional, random data)
        trained_model, training_info = trainer.train_hdbscan_with_fallbacks(
            embedding_matrix, params, []
        )
        
        if trained_model is not None:
            cluster_labels = trained_model.labels_
            
            # Calculate metrics
            metrics, metric_errors = trainer.calculate_clustering_metrics(embedding_matrix, cluster_labels)
            
            # Test quality validation with dataset size awareness
            quality_acceptable, quality_issues = trainer.validate_clustering_quality(metrics, len(embedding_matrix))
            
            print(f"🔍 CLUSTERING RESULTS:")
            print(f"   Clusters found: {metrics.get('n_clusters', 0)}")
            print(f"   Noise ratio: {metrics.get('noise_ratio', 0):.1%}")
            print(f"   Min cluster size: {metrics.get('min_cluster_size', 0)}")
            
            if 'silhouette_score' in metrics:
                print(f"   Silhouette score: {metrics['silhouette_score']:.3f}")
            
            if not quality_acceptable:
                print(f"\n⚠️ RESULT: Quality validation failed (but training continues)")
                print(f"   QUALITY ISSUES: {quality_issues}")
                print(f"   CONSEQUENCE: Model is saved but marked as low quality")
                print(f"   USER SEES: Warning messages about poor clustering quality")
            else:
                print(f"\n✅ RESULT: Quality validation passed")
                
    except Exception as e:
        print(f"💥 RESULT: Quality validation crashed")
        print(f"   ERROR: {str(e)}")

def demonstrate_save_failures():
    """Show what happens when saving fails"""
    
    print("\n" + "="*80)
    print("💾 PERSISTENCE STAGE FAILURE EXAMPLES")
    print("="*80)
    
    print("\n--- Common Save Failure Scenarios ---")
    
    save_failure_scenarios = [
        {
            "scenario": "Insufficient Disk Space",
            "consequence": "Model trains successfully but cannot be saved",
            "user_impact": "Training appears successful but no artifacts are saved",
            "error_example": "OSError: No space left on device"
        },
        {
            "scenario": "Permission Denied",
            "consequence": "Model trains but save fails due to file permissions",
            "user_impact": "Training succeeds but results are lost",
            "error_example": "PermissionError: Access denied to output directory"
        },
        {
            "scenario": "Corrupted Model Object",
            "consequence": "Model cannot be serialized properly",
            "user_impact": "Training appears successful but model is unusable",
            "error_example": "PicklingError: Cannot serialize model object"
        },
        {
            "scenario": "Invalid Output Path",
            "consequence": "Cannot create output directory or files",
            "user_impact": "Training succeeds but results cannot be stored",
            "error_example": "FileNotFoundError: Invalid path specified"
        }
    ]
    
    for scenario in save_failure_scenarios:
        print(f"\n--- {scenario['scenario']} ---")
        print(f"❌ CONSEQUENCE: {scenario['consequence']}")
        print(f"👤 USER IMPACT: {scenario['user_impact']}")
        print(f"🔍 ERROR EXAMPLE: {scenario['error_example']}")
        print(f"🛡️ GUARDRAIL RESPONSE: Error logged, training marked as 'partially successful'")

def run_comprehensive_failure_demonstration():
    """Run all failure demonstrations"""
    
    print("🛡️ COMPREHENSIVE TRAINING FAILURE DEMONSTRATION")
    print("This shows exactly what happens when each validation stage fails")
    print("=" * 100)
    
    try:
        demonstrate_data_validation_failures()
        demonstrate_parameter_validation_failures() 
        demonstrate_training_failures()
        demonstrate_quality_validation_failures()
        demonstrate_save_failures()
        
        print("\n" + "="*80)
        print("📋 SUMMARY OF FAILURE BEHAVIORS")
        print("="*80)
        
        failure_summary = {
            "🔍 Data Validation": {
                "behavior": "STOPS training immediately",
                "user_impact": "Clear error message, no resources wasted",
                "typical_errors": ["Insufficient samples", "Invalid embeddings", "Memory limits"]
            },
            "⚙️ Parameter Validation": {
                "behavior": "STOPS before training starts", 
                "user_impact": "Parameter correction guidance provided",
                "typical_errors": ["Invalid min_cluster_size", "Unsupported metrics", "Parameter conflicts"]
            },
            "🧮 Preprocessing": {
                "behavior": "CONTINUES with warnings",
                "user_impact": "Training proceeds with degraded preprocessing",
                "typical_errors": ["Scaling failures", "PCA failures"]
            },
            "🎯 Training": {
                "behavior": "TRIES fallbacks, then fails if all fail",
                "user_impact": "Multiple attempts before giving up",
                "typical_errors": ["No clusters found", "Too much noise", "Convergence failures"]
            },
            "📊 Quality Validation": {
                "behavior": "CONTINUES but marks as low quality",
                "user_impact": "Model saved but flagged as poor quality",
                "typical_errors": ["Poor silhouette score", "Unbalanced clusters", "Too few clusters"]
            },
            "💾 Persistence": {
                "behavior": "LOGS error but training considered successful",
                "user_impact": "Manual intervention needed to save results",
                "typical_errors": ["Disk space", "Permissions", "Serialization errors"]
            }
        }
        
        for stage, info in failure_summary.items():
            print(f"\n{stage}:")
            print(f"  Behavior: {info['behavior']}")
            print(f"  User Impact: {info['user_impact']}")
            print(f"  Typical Errors: {', '.join(info['typical_errors'])}")
        
        print(f"\n🎯 KEY INSIGHT: The system uses a 'fail-fast' approach for critical issues")
        print(f"   but 'fail-gracefully' for quality and persistence issues.")
        
    except Exception as e:
        print(f"\n💥 Demonstration failed: {str(e)}")

if __name__ == "__main__":
    run_comprehensive_failure_demonstration()