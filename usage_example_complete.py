# filepath: c:\Users\Bachn\OneDrive\Desktop\my_projects\hdbscan\usage_example_complete.py
"""
Complete usage example for the enhanced preprocessing pipeline with embedding support.
Shows how to handle both summarization and embedding failures.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Setup logging
from logging_setup import setup_detailed_logging
setup_detailed_logging(logging.INFO)

# Import pipeline components
from preprocessing.orchestrator import PreprocessingOrchestrator
from config import Config  # Assume you have a config module

def run_complete_preprocessing_example():
    """Example of running the complete preprocessing pipeline"""
    
    print("="*80)
    print("COMPLETE PREPROCESSING PIPELINE EXAMPLE")
    print("="*80)
    
    # Load your incident data
    # df = pd.read_csv('your_incidents.csv')  # Replace with your data
    
    # For demonstration, create sample data
    sample_data = {
        'number': [f'INC{i:07d}' for i in range(1000, 1100)],
        'short_description': [
            'SharePoint site not loading' if i % 10 == 0 
            else f'Application error {i}' for i in range(100)
        ],
        'description': [
            'Users cannot access SharePoint site. Getting timeout errors.' * (50 if i % 15 == 0 else 1)  # Some very long
            for i in range(100)
        ],
        'business_service': [
            'SharePoint Online' if i % 5 == 0 
            else f'Application Service {i%10}' for i in range(100)
        ]
    }
    df = pd.DataFrame(sample_data)
    
    print(f"Input data: {len(df)} incidents")
    
    # Initialize the orchestrator
    config = Config()  # Your configuration
    orchestrator = PreprocessingOrchestrator(config)
    
    # Run the complete pipeline
    print("\nStarting complete preprocessing pipeline...")
    
    try:
        summaries, embedding_matrix, valid_indices, stats = orchestrator.run_complete_pipeline(
            df=df,
            summarization_batch_size=20,      # Larger batch for summarization
            embedding_batch_size=50,          # Even larger batch for embedding
            use_batch_embedding_api=True      # Use batch API for efficiency
        )
        
        print("\n" + "="*80)
        print("PIPELINE RESULTS")
        print("="*80)
        
        print(f"✅ Successfully processed: {len(valid_indices)}/{len(df)} incidents")
        print(f"📊 Embedding matrix shape: {embedding_matrix.shape}")
        print(f"🎯 Overall success rate: {stats['overall_pipeline']['overall_success_rate']:.1f}%")
        
        # Show stage-by-stage breakdown
        print(f"\nStage Breakdown:")
        print(f"  📝 Summarization: {stats['summarization']['successful_summarizations']}/{len(df)} "
              f"({stats['overall_pipeline']['summarization_success_rate']:.1f}%)")
        print(f"  🔢 Embedding: {stats['embedding']['successful_embeddings']}/{stats['embedding']['summaries_available']} "
              f"({stats['overall_pipeline']['embedding_success_rate']:.1f}%)")
        
        # Show failure details if any
        if stats['overall_pipeline']['total_failures']['total_failed_incidents'] > 0:
            print(f"\n❌ Failures:")
            print(f"  📝 Summarization failures: {stats['overall_pipeline']['total_failures']['summarization_failures']}")
            print(f"  🔢 Embedding failures: {stats['overall_pipeline']['total_failures']['embedding_failures']}")
            
            # Show failure breakdown
            if stats['summarization'].get('failure_breakdown'):
                print(f"\n  Summarization failure types:")
                for error_type, count in stats['summarization']['failure_breakdown'].items():
                    print(f"    - {error_type}: {count}")
            
            if stats['embedding'].get('failure_breakdown'):
                print(f"\n  Embedding failure types:")
                for error_type, count in stats['embedding']['failure_breakdown'].items():
                    print(f"    - {error_type}: {count}")
        
        # Get comprehensive failure report
        failure_report = orchestrator.get_failed_incidents_comprehensive_report()
        
        if failure_report['pipeline_summary']['all_failed_incidents']:
            print(f"\n🔍 Failed incident numbers (first 10): ")
            failed_incidents = failure_report['pipeline_summary']['all_failed_incidents'][:10]
            print(f"  {failed_incidents}")
            if len(failure_report['pipeline_summary']['all_failed_incidents']) > 10:
                print(f"  ... and {len(failure_report['pipeline_summary']['all_failed_incidents']) - 10} more")
        
        # Prepare data for clustering
        if len(embedding_matrix) > 0:
            print(f"\n🎯 Data ready for clustering:")
            print(f"  - {len(valid_indices)} incidents with embeddings")
            print(f"  - {embedding_matrix.shape[1]} dimensional embeddings")
            print(f"  - Valid incident indices: {list(valid_indices[:5])}{'...' if len(valid_indices) > 5 else ''}")
            
            # Example: Ready to pass to clustering algorithm
            print(f"\n✅ Ready to proceed with HDBSCAN clustering!")
            
        else:
            print(f"\n❌ No embeddings available - cannot proceed with clustering")
        
        return summaries, embedding_matrix, valid_indices, stats
        
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        print(f"\n❌ Pipeline failed: {e}")
        return None, None, None, None

def demonstrate_embedding_failure_scenarios():
    """Demonstrate specific embedding failure scenarios"""
    
    print("\n" + "="*80)
    print("EMBEDDING FAILURE SCENARIO DEMONSTRATIONS")
    print("="*80)
    
    # Scenario 1: Mix of valid and invalid summaries
    print("\n1. Testing embedding with mixed summary quality...")
    
    # Simulate summarization results with some failures
    mixed_summaries = pd.Series({
        0: "SharePoint Online experiencing connectivity issues",
        1: None,  # Failed summarization
        2: "",    # Empty summary
        3: "SQL Server database connection timeout errors",
        4: "Very short",  # Too short for meaningful embedding
        5: "Application server showing high CPU usage and slow response times"
    })
    
    config = Config()
    orchestrator = PreprocessingOrchestrator(config)
    
    try:
        embeddings, stats = orchestrator.embedding_processor.process_embeddings_batch(
            mixed_summaries,
            batch_size=10
        )
        
        print(f"   Results: {stats['successful_embeddings']}/{stats['summaries_available']} embeddings created")
        print(f"   Summarization failures: {stats['summarization_failures']}")
        print(f"   Embedding failures: {stats['embedding_failures']}")
        
        if stats['failure_breakdown']:
            print("   Embedding failure types:")
            for error_type, count in stats['failure_breakdown'].items():
                print(f"     - {error_type}: {count}")
        
    except Exception as e:
        print(f"   Error: {e}")

def save_results_for_clustering(embedding_matrix, valid_indices, output_path="clustering_data.npz"):
    """Save preprocessing results for later clustering"""
    
    if len(embedding_matrix) > 0:
        np.savez_compressed(
            output_path,
            embeddings=embedding_matrix,
            indices=valid_indices.values,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"\n💾 Saved clustering data to {output_path}")
        print(f"   - Embeddings shape: {embedding_matrix.shape}")
        print(f"   - Valid indices count: {len(valid_indices)}")
        
        # Show how to load it back
        print(f"\n📖 To load for clustering:")
        print(f"   data = np.load('{output_path}')")
        print(f"   embeddings = data['embeddings']")
        print(f"   indices = data['indices']")
    else:
        print(f"\n❌ No embeddings to save - all processing failed")

def show_example_log_output():
    """Show what the log output looks like during processing"""
    
    print("\n" + "="*80)
    print("EXAMPLE LOG OUTPUT")
    print("="*80)
    
    example_logs = """
2024-01-15 14:30:15 | INFO     | Starting complete preprocessing pipeline for 100 incidents
2024-01-15 14:30:15 | INFO     | ============================================================
2024-01-15 14:30:15 | INFO     | STAGE 1: SUMMARIZATION
2024-01-15 14:30:15 | INFO     | ============================================================
2024-01-15 14:30:15 | INFO     | Starting summarization pipeline: 100 incidents, batch_size=20
2024-01-15 14:30:15 | INFO     | Processing batch 1/5 - incidents 1 to 20
2024-01-15 14:30:35 | INFO     | Batch 1 complete: 18/20 successful (90.0%) in 20.3s
2024-01-15 14:30:35 | WARNING  | Batch 1 failures: ['INC0001002', 'INC0001007']
2024-01-15 14:31:15 | INFO     | Summarization complete: 92/100 successful (92.0%)
2024-01-15 14:31:15 | INFO     | ============================================================
2024-01-15 14:31:15 | INFO     | STAGE 2: EMBEDDING
2024-01-15 14:31:15 | INFO     | ============================================================
2024-01-15 14:31:15 | INFO     | Starting embedding pipeline: 92 summaries, batch_size=50
2024-01-15 14:31:16 | INFO     | Processing batch 1/2 - summaries 1 to 50
2024-01-15 14:31:18 | INFO     | Batch 1 complete: 48/50 successful (96.0%) in 2.1s
2024-01-15 14:31:20 | INFO     | Embedding complete: 90/92 successful embeddings (97.8% of valid summaries)
2024-01-15 14:31:20 | INFO     | Overall pipeline success: 90/100 incidents have embeddings (90.0% of original)
2024-01-15 14:31:20 | INFO     | ============================================================
2024-01-15 14:31:20 | INFO     | PIPELINE COMPLETE - SUMMARY
2024-01-15 14:31:20 | INFO     | ============================================================
2024-01-15 14:31:20 | INFO     | Overall Results:
2024-01-15 14:31:20 | INFO     |   - Total incidents processed: 100
2024-01-15 14:31:20 | INFO     |   - Incidents ready for clustering: 90 (90.0%)
2024-01-15 14:31:20 | INFO     | Stage Success Rates:
2024-01-15 14:31:20 | INFO     |   - Summarization: 92.0%
2024-01-15 14:31:20 | INFO     |   - Embedding: 97.8%
2024-01-15 14:31:20 | INFO     |   - Overall pipeline: 90.0%
2024-01-15 14:31:20 | INFO     | Clustering Data Ready:
2024-01-15 14:31:20 | INFO     |   - Embedding matrix shape: (90, 1536)
    """
    
    print(example_logs)

if __name__ == "__main__":
    # Show example log output
    show_example_log_output()
    
    # Run the complete example
    summaries, embedding_matrix, valid_indices, stats = run_complete_preprocessing_example()
    
    # Demonstrate embedding failure scenarios
    demonstrate_embedding_failure_scenarios()
    
    # Save results if successful
    if embedding_matrix is not None and len(embedding_matrix) > 0:
        save_results_for_clustering(embedding_matrix, valid_indices)
    
    print(f"\n🎉 Example complete! Check the log file for detailed processing information.")