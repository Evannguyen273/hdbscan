#!/usr/bin/env python3
"""
Example: Test preprocessing pipeline with your specific BigQuery data source

This example demonstrates:
1. Loading data from enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev
2. Processing the three text columns: description, short_description, business_service
3. Cleaning text (removing special characters, emails)
4. Combining the three columns
5. Using LLM to summarize into 30-word combined_incidents_summary
6. Format: "what issues, which application got affected"
"""

import asyncio
import logging
from datetime import datetime, timedelta
from config.config import get_config
from pipeline.preprocessing_pipeline import PreprocessingPipeline

# Configure logging to see processing details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_test.log'),
        logging.StreamHandler()
    ]
)

async def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with your specific requirements"""
    
    print("=== TESTING PREPROCESSING PIPELINE ===")
    print("Data Source: enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev")
    print("Processing: description + short_description + business_service")
    print("Output: 30-word combined_incidents_summary")
    print("Format: what issues, which application got affected")
    print()
    
    # Load configuration
    config = get_config()
    
    # Verify configuration
    print("Configuration Check:")
    print(f"✓ BigQuery source table: {config.bigquery.tables['raw_incidents']}")
    print(f"✓ Text columns to process: {config.preprocessing.text_columns_to_process}")
    print(f"✓ Summary column name: {config.preprocessing.summary_column_name}")
    print(f"✓ Max summary length: {config.preprocessing.summarization.max_summary_length} words")
    print(f"✓ Summarization enabled: {config.preprocessing.summarization.enabled}")
    print()
    
    # Initialize preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline(config)
    
    # For testing, let's create sample data that matches your structure
    # In production, this would come from BigQuery
    sample_data = create_sample_incident_data()
    
    print(f"Processing {len(sample_data)} sample incidents...")
    print()
    
    # Run preprocessing
    results = await preprocessing_pipeline.process_for_training(sample_data)
    
    # Display results
    print("=== PREPROCESSING RESULTS ===")
    
    if 'All_Centers' in results:
        result = results['All_Centers']
        
        print(f"Status: {result['status']}")
        print(f"Total incidents: {result['total_input_incidents']}")
        print(f"Successfully processed: {result['incidents_processed']}")
        print(f"Success rate: {result.get('text_processing_stats', {}).get('success_rate', 0):.1f}%")
        print()
        
        if result['status'] == 'success' and 'summaries' in result:
            print("Sample Combined Summaries:")
            print("-" * 50)
            
            # Show first few summaries
            summaries = result['summaries']
            incident_data = result['incident_data']
            
            for i, (idx, summary) in enumerate(summaries.head(3).items()):
                if pd.notna(summary):
                    original_row = incident_data.loc[idx]
                    
                    print(f"Incident {i+1}:")
                    print(f"  Original Description: {original_row.get('description', 'N/A')[:100]}...")
                    print(f"  Original Short Desc: {original_row.get('short_description', 'N/A')[:50]}...")
                    print(f"  Business Service: {original_row.get('business_service', 'N/A')}")
                    print(f"  Combined Summary: {summary}")
                    print(f"  Word Count: {len(summary.split())} words")
                    print()
            
            print("=== VALIDATION ===")
            # Validate format
            valid_summaries = 0
            for summary in summaries.dropna():
                word_count = len(summary.split())
                if word_count <= 30:
                    valid_summaries += 1
            
            print(f"Summaries within 30 words: {valid_summaries}/{len(summaries.dropna())} ({valid_summaries/len(summaries.dropna())*100:.1f}%)")
            
        else:
            print(f"❌ Processing failed: {result.get('reason', 'Unknown error')}")
    else:
        print("❌ No results returned from preprocessing pipeline")

def create_sample_incident_data():
    """Create sample incident data matching your BigQuery structure"""
    import pandas as pd
    
    sample_incidents = [
        {
            'number': 'INC001234',
            'sys_created_on': '2024-01-15 10:30:00',
            'description': 'SharePoint site is not loading properly. Users unable to access documents. Error message appears when trying to navigate to team sites.',
            'short_description': 'SharePoint access issue',
            'business_service': 'SharePoint Online'
        },
        {
            'number': 'INC001235', 
            'sys_created_on': '2024-01-15 11:45:00',
            'description': 'Email server timeout when sending large attachments. SMTP error 554 occurs frequently. Multiple users affected across different departments.',
            'short_description': 'Email attachment sending failure',
            'business_service': 'Exchange Online'
        },
        {
            'number': 'INC001236',
            'sys_created_on': '2024-01-15 14:20:00', 
            'description': 'Database connection pool exhausted during peak hours. Application showing connection timeout errors. Performance degradation observed.',
            'short_description': 'Database connection issues',
            'business_service': 'SQL Server Database'
        },
        {
            'number': 'INC001237',
            'sys_created_on': '2024-01-15 16:10:00',
            'description': 'Web application returning HTTP 500 errors intermittently. Load balancer health checks failing. Backend services unresponsive.',
            'short_description': 'Web app HTTP 500 errors',
            'business_service': 'Customer Portal'
        },
        {
            'number': 'INC001238',
            'sys_created_on': '2024-01-15 17:30:00',
            'description': 'Network connectivity lost between data centers. VPN tunnel down. Remote site cannot access central resources.',
            'short_description': 'Inter-site network connectivity down',
            'business_service': 'WAN Network'
        }
    ]
    
    return pd.DataFrame(sample_incidents)

if __name__ == "__main__":
    import pandas as pd
    
    # Run the test
    try:
        asyncio.run(test_preprocessing_pipeline())
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        logging.error("Test failed", exc_info=True)