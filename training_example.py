# filepath: c:\Users\Bachn\OneDrive\Desktop\my_projects\hdbscan\training_example.py
"""
Complete training pipeline example for cumulative HDBSCAN approach.
Shows how to train clustering models using the updated pipeline architecture.
"""

import pandas as pd
import numpy as np
import logging
import asyncio
from datetime import datetime
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Import updated pipeline components
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.training_pipeline import TrainingPipeline
from config.config import get_config, get_current_quarter

async def run_complete_training_cycle_example():
    """Example of running a complete training cycle for cumulative HDBSCAN approach"""
    
    print("="*80)
    print("COMPLETE TRAINING CYCLE EXAMPLE")
    print("="*80)
    
    # Initialize the pipeline orchestrator
    config = get_config()
    orchestrator = PipelineOrchestrator(config)
    
    # Set training parameters
    current_year = datetime.now().year
    current_quarter = get_current_quarter()
    
    print(f"Training for: {current_year} {current_quarter}")
    print(f"Training window: {config.training.training_window_months} months")
    
    # Run the complete training cycle
    print("\nStarting complete cumulative training cycle...")
    
    try:
        training_results = await orchestrator.run_training_cycle(
            year=current_year,
            quarter=current_quarter,
            force_retrain=False  # Skip if model already exists
        )
        
        if training_results["status"] == "success":
            print("\n" + "="*80)
            print("TRAINING CYCLE SUCCESSFUL!")
            print("="*80)
            
            print(f"📊 Training Overview:")
            print(f"  - Version: {training_results['version']}")
            print(f"  - Duration: {training_results['training_duration_seconds']/60:.1f} minutes")
            print(f"  - Models Trained: {training_results['models_trained']}")
            print(f"  - Tech Centers: {len(training_results['tech_centers'])}")
            
            print(f"\n🎯 Tech Centers Processed:")
            for tech_center in training_results['tech_centers']:
                print(f"  - {tech_center}")
            
            print(f"\n💾 Storage Results:")
            storage_results = training_results['storage_results']
            successful_storage = sum(1 for result in storage_results.values() 
                                   if result.get('status') == 'success')
            print(f"  - Successfully stored: {successful_storage}/{len(storage_results)} models")
            
        elif training_results["status"] == "skipped":
            print(f"\n⏭️ Training skipped: {training_results.get('reason', 'Model already exists')}")
            
        else:
            print("\n" + "="*80)
            print("TRAINING CYCLE FAILED!")
            print("="*80)
            print(f"❌ Error: {training_results.get('error', 'Unknown error')}")
            print(f"Duration: {training_results.get('training_duration_seconds', 0)/60:.1f} minutes")
        
        return training_results
        
    except Exception as e:
        logging.error("Training cycle failed: %s", e)
        print(f"\n❌ Training cycle failed: {e}")
        return None

async def run_prediction_cycle_example():
    """Example of running prediction cycle for real-time classification"""
    
    print("\n" + "="*80)
    print("PREDICTION CYCLE EXAMPLE")
    print("="*80)
    
    # Initialize the pipeline orchestrator
    config = get_config()
    orchestrator = PipelineOrchestrator(config)
    
    # Run prediction cycle for all tech centers
    print("Starting prediction cycle for all tech centers...")
    
    try:
        prediction_results = await orchestrator.run_prediction_cycle()
        
        if prediction_results["status"] == "success":
            print("\n✅ Prediction cycle successful!")
            print(f"📊 Results:")
            print(f"  - Duration: {prediction_results['prediction_duration_seconds']:.1f} seconds")
            print(f"  - Incidents Processed: {prediction_results['incidents_processed']}")
            print(f"  - Tech Centers: {prediction_results['tech_centers_processed']}")
            print(f"  - Predictions Stored: {prediction_results['predictions_stored']}")
            
        else:
            print(f"\n❌ Prediction cycle failed: {prediction_results.get('error', 'Unknown error')}")
        
        return prediction_results
        
    except Exception as e:
        logging.error("Prediction cycle failed: %s", e)
        print(f"\n❌ Prediction cycle failed: {e}")
        return None

async def demonstrate_single_tech_center_training():
    """Demonstrate training for a single tech center"""
    
    print("\n" + "="*80)
    print("SINGLE TECH CENTER TRAINING EXAMPLE")
    print("="*80)
    
    config = get_config()
    training_pipeline = TrainingPipeline(config)
    
    # Get first tech center from config
    tech_centers = config.tech_centers
    if not tech_centers:
        print("❌ No tech centers configured")
        return None
    
    tech_center = tech_centers[0]
    print(f"Training model for: {tech_center}")
    
    # Create sample preprocessing data (in real scenario, this comes from preprocessing pipeline)
    sample_data = create_sample_preprocessing_data()
    
    # Set version info
    current_year = datetime.now().year
    current_quarter = get_current_quarter()
    version = f"{current_year}_{current_quarter}"
    
    try:
        training_results = await training_pipeline.train_tech_center(
            tech_center=tech_center,
            preprocessing_data=sample_data,
            version=version,
            year=current_year,
            quarter=current_quarter
        )
        
        if training_results["status"] == "success":
            print(f"\n✅ Training successful for {tech_center}")
            print(f"📊 Results:")
            print(f"  - Duration: {training_results['training_duration_seconds']/60:.1f} minutes")
            print(f"  - Incidents Trained: {training_results['incidents_trained']}")
            print(f"  - Model Version: {training_results['version']}")
            
            # Show model details
            model_data = training_results['model_data']
            print(f"\n🎯 Model Details:")
            print(f"  - Models Created: {len(model_data['models'])}")
            print(f"  - Domain Count: {len(model_data.get('domain_info', {}))}")
            
            for domain, info in model_data.get('domain_info', {}).items():
                print(f"    - {domain}: {info['incident_count']} incidents, {info['cluster_count']} clusters")
            
        else:
            print(f"\n❌ Training failed for {tech_center}: {training_results.get('error', 'Unknown error')}")
        
        return training_results
        
    except Exception as e:
        logging.error("Single tech center training failed: %s", e)
        print(f"\n❌ Training failed: {e}")
        return None

async def run_single_tech_center_2024(tech_center_name: str = None):
    """Run clustering for a single tech center for the year 2024"""
    
    print("="*80)
    print("SINGLE TECH CENTER 2024 CLUSTERING")
    print("="*80)
    
    config = get_config()
    training_pipeline = TrainingPipeline(config)
    
    # Use provided tech center or first from config
    if tech_center_name:
        tech_center = tech_center_name
    else:
        tech_centers = config.tech_centers
        if not tech_centers:
            print("❌ No tech centers configured")
            return None
        tech_center = tech_centers[0]
    
    print(f"Training clustering model for: {tech_center}")
    print(f"Training period: Full Year 2024")
    
    # TODO: Replace with real data fetch from BigQuery
    # In real implementation, you would fetch 2024 data like this:
    # from data_access.bigquery_client import BigQueryClient
    # bigquery_client = BigQueryClient(config)
    # raw_data = bigquery_client.fetch_incidents_by_tech_center_and_year(tech_center, 2024)
    # preprocessing_data = await preprocessing_pipeline.process_for_training(raw_data)
    
    # For now, create sample data
    sample_data = create_sample_preprocessing_data()
    
    try:
        training_results = await training_pipeline.train_tech_center(
            tech_center=tech_center,
            preprocessing_data=sample_data,
            version="2024_full_year",
            year=2024,
            quarter="full_year"  # Special indicator for full year
        )
        
        if training_results["status"] == "success":
            print(f"\n✅ 2024 Clustering successful for {tech_center}")
            print(f"📊 Results:")
            print(f"  - Training Duration: {training_results['training_duration_seconds']/60:.1f} minutes")
            print(f"  - Incidents Processed: {training_results['incidents_trained']}")
            print(f"  - Model Version: {training_results['version']}")
            
            # Show detailed clustering results
            model_data = training_results['model_data']
            print(f"\n🎯 Clustering Details:")
            print(f"  - Total Models Created: {len(model_data['models'])}")
            
            if 'domain_info' in model_data:
                print(f"  - Domains Processed: {len(model_data['domain_info'])}")
                for domain, info in model_data['domain_info'].items():
                    clusters = info.get('cluster_count', 'N/A')
                    incidents = info.get('incident_count', 'N/A')
                    print(f"    - {domain}: {incidents} incidents → {clusters} clusters")
            
            print(f"\n💾 Storage:")
            print(f"  - Model stored in Azure Blob Storage")
            print(f"  - Model registry updated in BigQuery")
            print(f"  - Model ID: {training_results.get('model_id', 'N/A')}")
            
        else:
            print(f"\n❌ 2024 Clustering failed for {tech_center}")
            print(f"Error: {training_results.get('error', 'Unknown error')}")
        
        return training_results
        
    except Exception as e:
        logging.error("2024 clustering failed for %s: %s", tech_center, e)
        print(f"\n❌ Clustering failed: {e}")
        return None

def create_sample_preprocessing_data() -> Dict:
    """Create sample preprocessing data for demonstration"""
    
    # Create sample incident data
    incident_data = pd.DataFrame({
        'incident_id': [f'INC{i:07d}' for i in range(1000, 1100)],
        'short_description': [
            'SharePoint site not loading' if i % 10 == 0 
            else f'Application error {i}' for i in range(100)
        ],
        'description': [
            'Users cannot access SharePoint site. Getting timeout errors.' * (2 if i % 15 == 0 else 1)
            for i in range(100)
        ],
        'business_service': [
            'SharePoint Online' if i % 5 == 0 
            else f'Service {i%3}' for i in range(100)
        ]
    })
    
    # Create sample embeddings (1536 dimensions like OpenAI)
    np.random.seed(42)  # For reproducible results
    embeddings = np.random.normal(0, 1, (100, 1536)).astype(np.float32)
    
    # Add some clustering structure
    for i in range(0, 100, 20):
        cluster_center = np.random.normal(0, 2, 1536)
        embeddings[i:i+20] += cluster_center * 0.5
    
    return {
        'embeddings': embeddings,
        'incident_data': incident_data
    }

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

async def demonstrate_scheduled_training():
    """Demonstrate scheduled training functionality"""
    
    print("\n" + "="*80)
    print("SCHEDULED TRAINING EXAMPLE")
    print("="*80)
    
    config = get_config()
    orchestrator = PipelineOrchestrator(config)
    
    print("Checking if training is due based on schedule...")
    print(f"Training frequency: {config.training.frequency}")
    print(f"Training months: {config.training.schedule['months']}")
    
    try:
        training_results = await orchestrator.run_scheduled_training()
        
        if training_results["status"] == "success":
            print("\n✅ Scheduled training completed successfully!")
            print(f"📊 Results:")
            print(f"  - Version: {training_results['version']}")
            print(f"  - Duration: {training_results['training_duration_seconds']/60:.1f} minutes")
            print(f"  - Models Trained: {training_results['models_trained']}")
            
        elif training_results["status"] == "skipped":
            reason = training_results.get("reason", "unknown")
            if reason == "not_scheduled":
                print(f"\n⏭️ Training not due - next training in month {training_results.get('next_training_month', 'unknown')}")
            elif reason == "model_exists":
                print(f"\n⏭️ Model already exists for current period")
            else:
                print(f"\n⏭️ Training skipped: {reason}")
                
        else:
            print(f"\n❌ Scheduled training failed: {training_results.get('error', 'Unknown error')}")
        
        return training_results
        
    except Exception as e:
        logging.error("Scheduled training failed: %s", e)
        print(f"\n❌ Scheduled training failed: {e}")
        return None

def show_pipeline_status():
    """Show current pipeline status and configuration"""
    
    print("\n" + "="*80)
    print("PIPELINE STATUS AND CONFIGURATION")
    print("="*80)
    
    config = get_config()
    orchestrator = PipelineOrchestrator(config)
    
    status = orchestrator.get_pipeline_status()
    
    print("📊 Pipeline Configuration:")
    orchestrator_config = status['orchestrator_config']
    print(f"  - Training Frequency: {orchestrator_config['training_frequency']}")
    print(f"  - Training Window: {orchestrator_config['training_window_months']} months")
    print(f"  - Prediction Frequency: {orchestrator_config['prediction_frequency_minutes']} minutes")
    print(f"  - Tech Centers: {orchestrator_config['tech_centers_count']}")
    
    print(f"\n🎯 Tech Centers:")
    for i, tech_center in enumerate(config.tech_centers[:5], 1):  # Show first 5
        print(f"  {i}. {tech_center}")
    if len(config.tech_centers) > 5:
        print(f"  ... and {len(config.tech_centers) - 5} more")
    
    print(f"\n⚙️ Configuration Details:")
    print(f"  - HDBSCAN Parameters: {config.clustering.hdbscan}")
    print(f"  - Domain Grouping: {'Enabled' if config.clustering.domain_grouping['enabled'] else 'Disabled'}")
    if config.clustering.domain_grouping['enabled']:
        print(f"    - Max Domains: {config.clustering.domain_grouping['max_domains_per_tech_center']}")
        print(f"    - Min Incidents: {config.clustering.domain_grouping['min_incidents_per_domain']}")
    
    print(f"\n📈 Training Configuration:")
    print(f"  - Parallel Processing: {config.training.processing['parallel_tech_centers']}")
    print(f"  - Max Workers: {config.training.processing['max_workers']}")
    print(f"  - Timeout: {config.training.processing['timeout_hours']} hours")
    print(f"  - Batch Size: {config.training.processing['batch_size']}")

def show_expected_training_logs():
    """Show what the training logs look like"""
    
    print("\n" + "="*80)
    print("EXAMPLE TRAINING LOG OUTPUT")
    print("="*80)
    
    example_logs = """
2024-01-15 15:00:00 | INFO     | ================================================================================
2024-01-15 15:00:00 | INFO     | STARTING CUMULATIVE TRAINING CYCLE
2024-01-15 15:00:00 | INFO     | ================================================================================
2024-01-15 15:00:00 | INFO     | Version: 2024_q4
2024-01-15 15:00:00 | INFO     | Training window: 24 months
2024-01-15 15:00:00 | INFO     | Stage 1: Preparing cumulative dataset...
2024-01-15 15:00:05 | INFO     | Fetching cumulative data from 2022-12-15 to 2024-12-15 (24 months)
2024-01-15 15:00:30 | INFO     | Dataset prepared: 15,432 incidents across 25 tech centers
2024-01-15 15:00:30 | INFO     | Stage 2: Running preprocessing pipeline...
2024-01-15 15:05:45 | INFO     | Preprocessing completed: 14,891 embeddings ready (96.5% success rate)
2024-01-15 15:05:45 | INFO     | Stage 3: Running training for all tech centers...
2024-01-15 15:05:45 | INFO     | Starting training for tech center: BT-TC-Network Operations (586 incidents)
2024-01-15 15:06:12 | INFO     | Training completed for BT-TC-Network Operations: 8 clusters, 12% noise
2024-01-15 15:06:12 | INFO     | Starting training for tech center: BT-TC-Database Services (423 incidents)
2024-01-15 15:06:35 | INFO     | Training completed for BT-TC-Database Services: 6 clusters, 15% noise
2024-01-15 15:12:20 | INFO     | Training completed for all tech centers: 23/25 successful
2024-01-15 15:12:20 | INFO     | Stage 4: Storing models and updating registry...
2024-01-15 15:13:45 | INFO     | All models stored successfully in Azure Blob Storage
2024-01-15 15:13:45 | INFO     | Model registry updated in BigQuery
2024-01-15 15:13:45 | INFO     | Training cycle completed successfully in 13.8 minutes
    """
    
    print(example_logs)

async def main():
    """Main function to run all examples"""
    
    print("🚀 HDBSCAN Pipeline Training Examples")
    print("=" * 80)
    
    # Show pipeline configuration
    show_pipeline_status()
    
    # Show expected log output
    show_expected_training_logs()
    
    # Run training examples
    try:
        # Example 1: Complete training cycle
        await run_complete_training_cycle_example()
        
        # Example 2: Single tech center training
        await demonstrate_single_tech_center_training()
        
        # Example 3: Prediction cycle
        await run_prediction_cycle_example()
        
        # Example 4: Scheduled training
        await demonstrate_scheduled_training()
        
        print(f"\n🎉 All training examples completed!")
        print(f"📊 Check the logs and Azure Blob Storage for training results.")
        
    except Exception as e:
        logging.error("Training examples failed: %s", e)
        print(f"\n❌ Training examples failed: {e}")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "single_2024":
            # Run single tech center for 2024
            tech_center = sys.argv[2] if len(sys.argv) > 2 else None
            asyncio.run(run_single_tech_center_2024(tech_center))
            
        elif command == "full_cycle":
            # Run complete training cycle
            asyncio.run(run_complete_training_cycle_example())
            
        elif command == "prediction":
            # Run prediction cycle
            asyncio.run(run_prediction_cycle_example())
            
        elif command == "single_tc":
            # Run single tech center (current quarter)
            asyncio.run(demonstrate_single_tech_center_training())
            
        elif command == "status":
            # Show pipeline status
            show_pipeline_status()
            
        else:
            print("Available commands:")
            print("  single_2024 [tech_center_name]  - Run clustering for 1 tech center for 2024")
            print("  full_cycle                     - Run complete training cycle")
            print("  prediction                     - Run prediction cycle")
            print("  single_tc                      - Run single tech center (current quarter)")
            print("  status                         - Show pipeline status")
            print("\nExample: python training_example.py single_2024 'BT-TC-Network Operations'")
    else:
        # Run all examples
        asyncio.run(main())