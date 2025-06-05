#!/usr/bin/env python3
"""
Direct Preprocessing Runner with Command Line Arguments
Run preprocessing directly using existing components with flexible command line interface
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse  # Import argparse
import sys

# Add project to path
sys.path.append(str(Path(__file__).parent))

from config.config import load_config
from data_access.bigquery_client import BigQueryClient
from preprocessing.text_processing import TextProcessor

async def run_preprocessing_direct(tech_center: str, start_date: str, end_date: str):
    """
    Run preprocessing directly - no fancy scripts needed!
    
    Args:
        tech_center: e.g., 'US-EAST', 'EUROPE'
        start_date: '2024-01-01' 
        end_date: '2024-01-31'
    """
    
    print(f"🚀 Running preprocessing for {tech_center} from {start_date} to {end_date}")
    
    # Load your config
    config = load_config()
    
    # Initialize your existing components
    bq_client = BigQueryClient(config)
    text_processor = TextProcessor(config)
    
    # Convert dates
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    months_back = max(1, int((end_dt - start_dt).days / 30))
    
    # 1. Get incidents from BigQuery (using your existing method)
    print("📊 Fetching incidents...")
    incidents_df = bq_client.get_training_data_window(
        tech_center=tech_center,
        end_date=end_dt,
        months_back=months_back
    )
    
    if incidents_df.empty:
        print(f"❌ No incidents found for {tech_center}")
        return False
    
    print(f"✅ Found {len(incidents_df)} incidents")
    
    # 2. Process texts (using your existing text processor)
    print("🔄 Processing texts...")
    
    # Get combined text columns
    combined_texts = text_processor._combine_incident_columns(incidents_df)
    
    # Process texts in batches using your existing method
    processed_results = await text_processor.process_texts_for_training(
        texts=combined_texts,
        batch_size=10
    )
    
    processed_texts, valid_indices, stats = processed_results
    
    print(f"✅ Processed {len(processed_texts)} texts")
    print(f"📊 Stats: {stats}")
    
    # 3. Store results (using your existing method)
    print("💾 Storing results...")
    
    # Create result dataframe
    result_df = incidents_df.iloc[valid_indices].copy()
    result_df['processed_text'] = list(processed_texts.values())
    result_df['processing_timestamp'] = datetime.now()
    
    # Store using your existing method
    version = f"{tech_center}_{start_date}_{end_date}".replace('-', '_')
    success = bq_client.store_training_data(version, result_df)
    
    if success:
        print(f"✅ SUCCESS! Stored {len(result_df)} preprocessed incidents")
        print(f"📋 Version: {version}")
    else:
        print("❌ Failed to store results")
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing directly for a specific tech center and date range.")
    parser.add_argument("tech_center", type=str, help="Tech center to process (e.g., 'US-EAST')")
    parser.add_argument("start_date", type=str, help="Start date in YYYY-MM-DD format (e.g., '2024-01-01')")
    parser.add_argument("end_date", type=str, help="End date in YYYY-MM-DD format (e.g., '2024-01-31')")

    args = parser.parse_args()
    
    # Run it with command line arguments
    asyncio.run(run_preprocessing_direct(args.tech_center, args.start_date, args.end_date))