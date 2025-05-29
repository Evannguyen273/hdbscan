"""
Example: Updated Preprocessing Pipeline with Enhanced Error Reporting

This example shows how to use the updated preprocessing pipeline that:
1. Skips failed summarization attempts (no fallbacks)
2. Uses chunking strategy for long texts exceeding token limits
3. Reports all failed incident numbers at the end
4. Saves detailed failure reports for investigation
"""

import logging
from datetime import datetime
from config.config_loader import ConfigLoader
from pipeline.preprocessing_pipeline import PreprocessingPipeline

# Configure logging to see all messages including warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

def run_preprocessing_example():
    """Run preprocessing with enhanced error reporting"""
    
    # Load configuration
    config = ConfigLoader.load_config("config/config.yaml")
    
    # Initialize preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline(config)
    
    print("=== STARTING PREPROCESSING WITH NO-FALLBACK POLICY ===")
    print("Features:")
    print("- Skips failed summarizations (no fallbacks)")
    print("- Uses chunking for texts exceeding token limits")
    print("- Reports all failed incident numbers")
    print("- Saves detailed failure reports\n")
    
    # Run preprocessing for all tech centers
    results = preprocessing_pipeline.run_preprocessing_all_tech_centers()
    
    # Display results summary
    print("\n=== PREPROCESSING RESULTS SUMMARY ===")
    print(f"Tech Centers Processed: {results['total_tech_centers']}")
    print(f"Successful Tech Centers: {results['successful']}")
    print(f"Failed Tech Centers: {results['failed']}")
    print(f"Total Incidents Processed: {results['total_processed']}")
    print(f"Total Failed Incidents: {results['total_failed_incidents']}")
    
    # Show failed incidents if any
    if results['total_failed_incidents'] > 0:
        print(f"\n🚨 FAILED INCIDENTS REQUIRING INVESTIGATION:")
        print(f"Failed Incident Numbers: {results['failed_incident_numbers']}")
        print(f"\nNext Steps:")
        print(f"1. Check ServiceNow for these incident numbers")
        print(f"2. Review failure details in: preprocessing/failed_incidents/")
        print(f"3. Check logs for specific error patterns")
        
        # Show breakdown by tech center
        print(f"\nFailure Breakdown by Tech Center:")
        for tech_result in results['results']:
            if tech_result.get('failed_incidents'):
                tech_center = tech_result['tech_center']
                failed_count = len(tech_result['failed_incidents'])
                success_rate = tech_result.get('success_rate', 0)
                print(f"  {tech_center}: {failed_count} failed ({success_rate:.1f}% success rate)")
                print(f"    Failed incidents: {tech_result['failed_incidents']}")
    else:
        print(f"\n✅ All incidents processed successfully!")
    
    print(f"\nTotal Runtime: {results['runtime_seconds']:.2f} seconds")
    
    return results

def investigate_failed_incident(incident_number: str):
    """Helper function to investigate a specific failed incident"""
    
    print(f"\n=== INVESTIGATING FAILED INCIDENT: {incident_number} ===")
    
    # This would typically query your source data to get incident details
    # For demonstration, showing what should be checked
    
    investigation_steps = [
        "1. Check incident in ServiceNow:",
        f"   - Search for incident number: {incident_number}",
        f"   - Review short_description length and content",
        f"   - Review description length and content", 
        f"   - Check for special characters or encoding issues",
        "",
        "2. Check failure logs:",
        f"   - Look in preprocessing/failed_incidents/ for reports",
        f"   - Search for '{incident_number}' in recent log files",
        "",
        "3. Common failure reasons to check:",
        f"   - Rate limiting (429 errors)",
        f"   - Token limits exceeded (even with chunking)",
        f"   - Malformed text content",
        f"   - Network timeouts",
        f"   - Missing required fields",
        "",
        "4. Possible resolutions:",
        f"   - Wait and retry if rate limited",
        f"   - Manual text cleanup if content issues",
        f"   - Adjust batch size if consistent failures",
        f"   - Skip incident if content is fundamentally problematic"
    ]
    
    for step in investigation_steps:
        print(step)

def demo_chunking_strategy():
    """Demonstrate the chunking strategy for long texts"""
    
    print("\n=== CHUNKING STRATEGY DEMONSTRATION ===")
    
    # Simulate a very long incident description
    long_description = "System error occurred. " * 5000  # ~50k characters = ~12.5k tokens
    
    from preprocessing.text_processing import TextProcessor
    from config.config_loader import ConfigLoader
    
    config = ConfigLoader.load_config("config/config.yaml")
    text_processor = TextProcessor(config)
    
    # Check token estimation
    estimated_tokens = text_processor.estimate_tokens(long_description)
    print(f"Long description estimated tokens: {estimated_tokens}")
    
    # Show chunking in action
    chunks = text_processor.chunk_text_for_summarization(long_description, max_input_tokens=100000)
    print(f"Text split into {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = text_processor.estimate_tokens(chunk)
        print(f"  Chunk {i+1}: {chunk_tokens} tokens")
    
    print(f"\nTotal context window check:")
    max_output_tokens = 100
    total_input_tokens = sum(text_processor.estimate_tokens(chunk) for chunk in chunks)
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Max output tokens: {max_output_tokens}")
    print(f"Would exceed 128k limit: {total_input_tokens + max_output_tokens > 120000}")

if __name__ == "__main__":
    # Run the main preprocessing example
    results = run_preprocessing_example()
    
    # If there were failures, demonstrate investigation
    if results.get('failed_incident_numbers'):
        example_failed_incident = results['failed_incident_numbers'][0]
        investigate_failed_incident(example_failed_incident)
    
    # Show chunking strategy demo
    demo_chunking_strategy()