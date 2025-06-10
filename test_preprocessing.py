#!/usr/bin/env python3
"""
Direct Preprocessing Runner with Command Line Arguments
Run preprocessing directly using existing components with flexible command line interface
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import sys

# Add project to path
sys.path.append(str(Path(__file__).parent))

from config.config import load_config
from data_access.bigquery_client import BigQueryClient
from preprocessing.text_processing import TextProcessor

async def run_preprocessing_direct(start_date: str = None, end_date: str = None, quiet: bool = False, overwrite: bool = False):
    """
    Run preprocessing directly - no fancy scripts needed!

    Args:
        start_date: '2024-01-01'
        end_date: '2024-01-31'
        quiet: If True, suppress distribution details
        overwrite: If True, overwrite existing data in the table. If False, append to it.
    """
    # Default dates if not provided
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        # Default to 30 days before end date
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=30)
        start_date = start_dt.strftime('%Y-%m-%d')

    print(f"🚀 Running preprocessing from {start_date} to {end_date}")

    # Load your config
    config = load_config()

    # Initialize your existing components
    bq_client = BigQueryClient(config)
    text_processor = TextProcessor(config)

    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Adjust end date to include the entire day (23:59:59)
    end_dt = end_dt.replace(hour=23, minute=59, second=59)

    # Format dates for SQL (use isoformat for proper timestamp handling)
    sql_start_date = start_dt.isoformat()
    sql_end_date = end_dt.isoformat()

    # 1. Get incidents from BigQuery
    print("📊 Fetching incidents...")

    # Direct SQL approach for better control
    source_table = config.bigquery.tables.incident_source

    # Create a direct query that matches fields needed
    direct_query = f"""
    SELECT
        number,
        sys_created_on,
        description,
        short_description,
        business_service
    FROM `{source_table}`
    WHERE sys_created_on >= '{sql_start_date}'
    AND sys_created_on <= '{sql_end_date}'
    ORDER BY sys_created_on DESC
    """

    # Print query execution start time
    print(f"🔍 Executing BigQuery query at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    print(f"🔍 Query: {direct_query}")

    # Execute the query
    incidents_df = bq_client.client.query(direct_query).to_dataframe()

    # Print query completion
    print(f"✅ Query completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if incidents_df.empty:
        print(f"❌ No incidents found for the specified date range")
        return False

    # Show basic distribution stats if not in quiet mode
    if not quiet and 'business_service' in incidents_df.columns:
        service_counts = incidents_df['business_service'].value_counts()
        print("📊 Incidents by business service:")
        print(f"  - Top {min(10, len(service_counts))} of {len(service_counts)} business services found")

        # Only show top 10 in non-quiet mode
        for service, count in service_counts.head(10).items():
            print(f"  - {service}: {count}")

    print(f"✅ Found {len(incidents_df)} incidents")

    # 2. Process texts (using your existing text processor)
    print("🔄 Processing texts...")

    # Get combined text columns
    combined_texts = text_processor._combine_incident_columns(incidents_df)

    # Process texts in batches using your existing method with larger batch size for better performance
    processed_results = await text_processor.process_texts_for_training(
        texts=combined_texts,
        batch_size=50  # Increased from 10 to 50 for better performance
    )

    processed_texts, valid_indices, stats = processed_results

    print(f"✅ Processed {len(processed_texts)} texts")
    print(f"📊 Stats: {stats}")

    # Generate embeddings for the processed texts
    print("🧠 Generating embeddings...")
    embeddings = await text_processor._generate_embeddings_batch(processed_texts.tolist())
    print(f"✅ Generated {len(embeddings)} embeddings")

    # 3. Store results
    write_mode = "TRUNCATE" if overwrite else "APPEND"
    print(f"💾 Storing results (mode: {write_mode})...")

    # Create result dataframe matching preprocessed_incidents schema
    result_df = incidents_df.iloc[valid_indices].copy()

    # Ensure column names match schema
    if 'incident_number' in result_df.columns and 'number' not in result_df.columns:
        result_df = result_df.rename(columns={'incident_number': 'number'})
    if 'created_date' in result_df.columns and 'sys_created_on' not in result_df.columns:
        result_df = result_df.rename(columns={'created_date': 'sys_created_on'})

    # Add required fields for preprocessed_incidents schema
    result_df['combined_incidents_summary'] = processed_texts.tolist()
    result_df['created_timestamp'] = datetime.now()
    result_df['processing_version'] = f"v2.0.0_all_{start_date}_{end_date}"

    # Use actual embeddings instead of placeholder values
    result_df['embedding'] = embeddings

    # Select only the columns that match preprocessed_incidents schema
    final_columns = ['number', 'sys_created_on', 'combined_incidents_summary',
                    'embedding', 'created_timestamp', 'processing_version']
    result_df = result_df[final_columns]

    # Store using the BigQuery client method
    success = bq_client.store_preprocessed_incidents(result_df, overwrite=overwrite)

    # Update the watermark with the most recent incident timestamp
    if success:
        newest_incident_timestamp = incidents_df['sys_created_on'].max()
        newest_incident_id = incidents_df.loc[incidents_df['sys_created_on'].idxmax(), 'number']

        # Update the watermark table
        await bq_client.update_preprocessing_watermark(
            preprocessed_rows=len(result_df),
            timestamp=datetime.now(),  # Add timestamp parameter
            run_details={
                "start_date": datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%dT%H:%M:%S'),
                "end_date": end_dt.strftime('%Y-%m-%dT%H:%M:%S'),
                "rows_processed": len(result_df),
                "newest_incident_id": newest_incident_id,
                "newest_incident_timestamp": newest_incident_timestamp.strftime('%Y-%m-%dT%H:%M:%S')
            }
        )

        print(f"✅ SUCCESS! Stored {len(result_df)} preprocessed incidents ({write_mode} mode)")
        print(f"📋 Table: preprocessed_incidents")
        print(f"📋 Processing Version: {result_df['processing_version'].iloc[0]}")
        print(f"📋 Updated watermark with timestamp: {newest_incident_timestamp}")
    else:
        print("❌ Failed to store preprocessed results")

    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing for incidents in a specific date range.")
    parser.add_argument("start_date", type=str, help="Start date in YYYY-MM-DD format (e.g., '2024-01-01')")
    parser.add_argument("end_date", type=str, help="End date in YYYY-MM-DD format (e.g., '2024-01-31')")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress distribution details")
    parser.add_argument("--overwrite", "-o", action="store_true",
                      help="Overwrite existing data in BigQuery table. If not specified, data will be appended.")

    args = parser.parse_args()

    # Run it with command line arguments
    asyncio.run(run_preprocessing_direct(args.start_date, args.end_date, args.quiet, args.overwrite))