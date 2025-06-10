"""
Simplified preprocessing pipeline for hourly execution in Azure Functions
"""
import logging
import pandas as pd
import numpy as np
import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure OpenAI logging to suppress retry messages
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)

from config.config import get_config
from data_access.bigquery_client import BigQueryClient
from preprocessing.text_processing import TextProcessor
from preprocessing.embedding_generation import EmbeddingGenerator


class PreprocessingPipeline:
    """
    Simplified preprocessing pipeline for incremental incident processing.
    Designed for hourly execution in Azure Functions.
    """

    def __init__(self, config=None):
        """Initialize preprocessing pipeline with dependencies"""
        self.config = config if config is not None else get_config()
        self.bq_client = BigQueryClient(self.config)
        self.text_processor = TextProcessor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)

        # Set the source and watermark table names
        self.source_table = self.config.bigquery.tables.incident_source
        self.watermark_table = self.config.bigquery.tables.watermarks

        logging.info("Preprocessing pipeline initialized for hourly execution")

    async def run(self, start_date: str = None, end_date: str = None, overwrite: bool = False, overwrite_watermark: bool = False, quiet: bool = True) -> bool:
        """
        Run the preprocessing pipeline with watermark-based incremental processing

        Args:
            start_date: Optional override start date (YYYY-MM-DD)
            end_date: Optional override end date (YYYY-MM-DD)
            overwrite: If True, overwrite existing data in the destination table
            overwrite_watermark: If True, overwrite existing watermarks instead of appending
            quiet: If True, suppress detailed output (default: True)

        Returns:
            bool: True if successful, False otherwise
        """
        # Record the exact time when the query starts - this will be our time_trigger
        query_start_timestamp = datetime.now()
        logging.info(f"Starting preprocessing pipeline run at {query_start_timestamp}")

        # Ensure watermark table exists
        await self.bq_client.create_preprocessing_watermarks_table()

        # Determine the appropriate start and end timestamps
        start_time, end_time = await self._determine_time_window(start_date, end_date, query_start_timestamp)

        # Format query using the simplified template
        query = self._build_query_from_template(start_time, end_time)

        # Execute the query
        logging.info(f"Executing BigQuery query at {query_start_timestamp}")
        logging.info(f"Query: {query}")

        incidents_df = self.bq_client.client.query(query).to_dataframe()
        logging.info(f"Query completed at {datetime.now()}")

        if incidents_df.empty:
            logging.info("No new incidents to process")
            # Still update the watermark to record this run - using the query start timestamp
            await self._update_watermark(
                rows_processed=0,
                timestamp=query_start_timestamp,  # Use query start time for time_trigger
                run_details={
                    "start_date": start_time.strftime('%Y-%m-%d %H:%M:%S UTC'),  # Updated format with UTC
                    "end_date": end_time.strftime('%Y-%m-%d %H:%M:%S UTC'),      # Updated format with UTC
                    "rows_processed": 0,
                    "newest_incident_id": None,
                    "newest_incident_timestamp": None
                },
                overwrite=overwrite_watermark
            )
            return True

        logging.info(f"Processing {len(incidents_df)} incidents")

        # Track newest incident for logging
        newest_incident_timestamp = incidents_df['sys_created_on'].max()
        newest_incident_id = incidents_df.loc[incidents_df['sys_created_on'].idxmax(), 'number']

        # Process incidents
        processed_df = await self._process_incidents(incidents_df)
        if processed_df.empty:
            logging.error("Processing failed - no results")
            return False

        # Store processed results
        success = self.bq_client.store_preprocessed_incidents(processed_df, overwrite=overwrite)
        if not success:
            logging.error("Failed to store processed incidents")
            return False

        # Update watermark with the query start timestamp (not the current time)
        run_details = {
            "start_date": start_time.strftime('%Y-%m-%d %H:%M:%S UTC'),  # Updated format with UTC
            "end_date": end_time.strftime('%Y-%m-%d %H:%M:%S UTC'),      # Updated format with UTC
            "rows_processed": len(processed_df),
            "newest_incident_id": newest_incident_id,
            "newest_incident_timestamp": newest_incident_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')  # Updated format with UTC
        }

        await self._update_watermark(
            rows_processed=len(processed_df),
            timestamp=query_start_timestamp,  # Use query start time for time_trigger
            run_details=run_details,
            overwrite=overwrite_watermark
        )

        logging.info(f"Successfully processed {len(processed_df)} incidents")
        logging.info(f"Watermark updated with timestamp: {query_start_timestamp}")

        return True

    def _parse_json_safely(self, json_str):
        """Parse JSON string safely and return a dictionary."""
        if not json_str:
            return {}

        if not isinstance(json_str, str):
            # Already parsed
            return json_str if isinstance(json_str, dict) else {}

        try:
            import json
            result = json.loads(json_str)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logging.error(f"Failed to parse JSON: {str(e)}")
            return {}

    async def _calculate_hourly_time_window(self, current_timestamp: datetime) -> Tuple[datetime, datetime]:
        """
        Calculate time window for an hourly run when no dates are specified

        Args:
            current_timestamp: Current timestamp when the run was triggered

        Returns:
            Tuple of (start_time, end_time) as datetime objects
        """
        # Get latest watermark to find the last processed time
        latest_watermark = await self.bq_client.get_latest_preprocessing_watermark_with_details()

        if not latest_watermark:
            # If no previous run found, default to 1 hour ago from current time
            start_time = current_timestamp - timedelta(hours=1)
            end_time = current_timestamp
            logging.info(f"No previous watermark found. Using 1-hour window: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        try:
            # Parse end_date from the previous run
            raw_details = latest_watermark.get('run_details', '')
            print(f"🔍 DEBUG: Raw details from watermark for hourly window: {raw_details}")

            # Parse the JSON details
            if isinstance(raw_details, str):
                if raw_details.startswith('"') and raw_details.endswith('"'):
                    import json
                    raw_details = json.loads(raw_details[1:-1].replace('\\"', '"'))
                else:
                    import json
                    raw_details = json.loads(raw_details)

            # Get the end_date from details
            if isinstance(raw_details, dict) and 'end_date' in raw_details:
                end_date_str = raw_details['end_date']
                print(f"🔍 DEBUG: Found previous end_date: {end_date_str}")

                # Parse the previous end date
                prev_end_time = self._parse_timestamp(end_date_str)
                if prev_end_time:
                    # For hourly runs, start from the previous end timestamp
                    start_time = prev_end_time + timedelta(seconds=1)

                    # End time is 1 hour after start, but capped at current time
                    hourly_end_time = start_time + timedelta(hours=1)
                    end_time = min(hourly_end_time, current_timestamp)

                    print(f"📅 Hourly window: {start_time.isoformat()} to {end_time.isoformat()}")
                    return start_time, end_time
        except Exception as e:
            logging.error(f"Error calculating hourly window: {e}")
            # Fall back to default 1-hour window

        # Default to 1-hour window if we couldn't parse from watermark
        start_time = current_timestamp - timedelta(hours=1)
        end_time = current_timestamp
        logging.info(f"Using default 1-hour window: {start_time.isoformat()} to {end_time.isoformat()}")
        return start_time, end_time

    async def _determine_time_window(self, start_date: str, end_date: str, run_timestamp: datetime) -> Tuple[datetime, datetime]:
        """
        Determine the appropriate start and end timestamps for the query

        Args:
            start_date: Optional start date string (YYYY-MM-DD)
            end_date: Optional end date string (YYYY-MM-DD)
            run_timestamp: Current run timestamp

        Returns:
            Tuple of (start_time, end_time) as datetime objects
        """
        # If neither start_date nor end_date specified, use hourly window logic
        if start_date is None and end_date is None:
            return await self._calculate_hourly_time_window(run_timestamp)

        # Handle end time first (simpler logic)
        if end_date:
            # Convert to datetime and set to end of day
            end_time = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
            print(f"📅 End date for sys_created_on filter: {end_time.isoformat()} (from manual input)")
        else:
            # Use current time
            end_time = run_timestamp
            print(f"📅 End date for sys_created_on filter: {end_time.isoformat()} (current time)")

        # Handle start time (more complex - watermark logic)
        if start_date:
        # User specified a start date - simple case
            start_time = datetime.strptime(start_date, '%Y-%m-%d')
            print(f"📅 Start date for sys_created_on filter: {start_time.isoformat()} (from manual input)")
        else:
            # Complex watermark logic
            latest_watermark = await self.bq_client.get_latest_preprocessing_watermark_with_details()

            if latest_watermark:
                try:
                    # Get the raw run_details string
                    raw_details = latest_watermark.get('run_details', '')
                    print(f"🔍 DEBUG: Raw details from watermark: {raw_details}")

                    # Try simple JSON parsing first
                    try:
                        import json
                        # First, check if the string has quotes at beginning and end - this means it's a JSON string
                        # that has been double-quoted in the database
                        if isinstance(raw_details, str) and raw_details.startswith('"') and raw_details.endswith('"'):
                            # Remove the outer quotes and unescape inner quotes
                            clean_json = raw_details[1:-1].replace('\\"', '"')
                            details_dict = json.loads(clean_json)
                        else:
                            # Standard JSON parsing
                            details_dict = json.loads(raw_details) if isinstance(raw_details, str) else raw_details

                        # Now check if end_date is in the dictionary
                        if isinstance(details_dict, dict) and 'end_date' in details_dict:
                            end_date_str = details_dict['end_date']
                            print(f"🔍 DEBUG: Found end_date in parsed JSON: {end_date_str}")

                            # Parse the end_date and use it
                            end_time_from_details = self._parse_timestamp(end_date_str)
                            if end_time_from_details:
                                # Add 1 second to ensure no gaps
                                start_time = end_time_from_details + timedelta(seconds=1)
                                print(f"📅 Start date: {start_time.isoformat()} (from previous run's end_date + 1 second)")
                                return start_time, end_time
                    except Exception as json_error:
                        print(f"🔍 DEBUG: JSON parsing failed: {str(json_error)}, trying regex extraction")

                    # Fall back to regex extraction if JSON parsing failed
                    if isinstance(raw_details, str):
                        import re
                        # Updated pattern for escaped quotes in JSON string
                        # This works for both normal and escaped JSON strings
                        pattern = r'"end_date":\s*"([^"]+)"|\\\"end_date\\\":\\s*\\\"([^\\\"]+)\\\"'
                        matches = re.findall(pattern, raw_details)
                        if matches:
                            # Extract the first non-empty group from matches
                            for match_groups in matches:
                                for group in match_groups:
                                    if group:
                                        end_date_str = group
                                        print(f"🔍 DEBUG: Found end_date with regex: {end_date_str}")

                                        # Parse the end_date and use it
                                        end_time_from_details = self._parse_timestamp(end_date_str)
                                        if end_time_from_details:
                                            # Add 1 second to ensure no gaps
                                            start_time = end_time_from_details + timedelta(seconds=1)
                                            print(f"📅 Start date: {start_time.isoformat()} (from previous run's end_date + 1 second)")
                                            return start_time, end_time

                    # If we got here, we couldn't extract the end_date
                    raise ValueError("Could not extract end_date from watermark")

                except Exception as e:
                    # Instead of falling back, raise the error to fail execution
                    logging.error(f"Error getting start date from previous run: {str(e)}")
                    raise ValueError(f"Cannot determine start date from previous run: {str(e)}")

        return start_time, end_time

    def _build_query_from_template(self, start_time: datetime, end_time: datetime) -> str:
        """
        Build the query string from the template with the given time range

        Args:
            start_time: Start time as datetime object
            end_time: End time as datetime object

        Returns:
            str: Formatted query string
        """
        # Get query template from configuration
        query_template = self.config.bigquery.queries.incident_data_for_preprocessing

        # Format the query with the actual table and time range - use ISO 8601 format with T separator
        # to match the format of the original sys_created_on values
        query = query_template.format(
            source_table=self.source_table,
            start_time=start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            end_time=end_time.strftime('%Y-%m-%dT%H:%M:%S')
        )

        return query

    async def _process_incidents(self, incidents_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process incidents through text processing and embedding generation

        Returns:
            Processed DataFrame with summaries and embeddings
        """
        # Step 1: Combine text columns for processing
        combined_texts = self.text_processor._combine_incident_columns(incidents_df)
        logging.info(f"Combined {len(combined_texts)} text columns for processing")

        # Step 2: Process texts for summarization
        batch_size = self.config.preprocessing.summarization.get('batch_size', 10)
        logging.info(f"Processing text with batch size {batch_size}")

        processed_results = await self.text_processor.process_texts_for_training(
            texts=combined_texts,
            batch_size=batch_size
        )

        summaries, valid_indices, text_stats = processed_results

        if len(summaries) == 0:
            logging.error("Text processing failed - no valid summaries generated")
            return pd.DataFrame()

        logging.info(f"Successfully processed {len(summaries)} summaries")

        # Step 3: Generate embeddings
        embedding_batch_size = 50
        logging.info(f"Generating embeddings with batch size {embedding_batch_size}")

        embeddings, embedding_stats = await self.embedding_generator.generate_embeddings_batch(
            summaries.tolist(),
            batch_size=embedding_batch_size
        )

        if len(embeddings) == 0:
            logging.error("Embedding generation failed - no embeddings created")
            return pd.DataFrame()

        logging.info(f"Successfully generated {len(embeddings)} embeddings")

        # Step 4: Create final processed DataFrame
        processed_df = incidents_df.iloc[valid_indices].copy()
        processed_df['combined_incidents_summary'] = summaries.tolist()

        # Convert numpy 2D array to list of arrays if needed
        if isinstance(embeddings, np.ndarray) and embeddings.ndim > 1:
            embedding_list = [emb.tolist() for emb in embeddings]
            logging.info(f"Converting embeddings from shape {embeddings.shape} to list of {len(embedding_list)} arrays")
            processed_df['embedding'] = embedding_list
        else:
            # Already a list format
            processed_df['embedding'] = embeddings

        processed_df['created_timestamp'] = datetime.now()
        processed_df['processing_version'] = self.config.preprocessing.processing_version

        # Select only needed columns
        columns = ['number', 'sys_created_on', 'combined_incidents_summary',
                   'embedding', 'created_timestamp', 'processing_version']
        return processed_df[columns]

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse a timestamp string into a datetime object, supporting multiple formats

        Args:
            timestamp_str: Timestamp string to parse

        Returns:
            Optional[datetime]: Parsed datetime object, or None if parsing failed
        """
        # Check if timestamp has UTC suffix and remove it for parsing
        if timestamp_str.endswith(' UTC'):
            timestamp_str = timestamp_str[:-4]  # Remove ' UTC'

        try:
            # Try parsing with T separator and milliseconds (ISO format)
            return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            try:
                # Try parsing with T separator, no milliseconds (ISO format)
                return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                try:
                    # Try parsing with space separator and milliseconds
                    return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    try:
                        # Try parsing with space separator, no milliseconds
                        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except Exception as e:
                        logging.error(f"Failed to parse timestamp {timestamp_str}: {str(e)}")
                        return None

    async def _update_watermark(self, rows_processed: int, timestamp: datetime, run_details: dict, overwrite: bool = False):
        """
        Update the preprocessing watermark with the latest run information

        Args:
            rows_processed: Number of rows processed in the run
            timestamp: Timestamp of the run (time_trigger)
            run_details: Additional details about the run (e.g., start_date, end_date)
            overwrite: If True, overwrite the existing watermark row instead of appending
        """
        # Simplify: just ensure we have a naive datetime without timezone info
        if timestamp.tzinfo is not None:
            timestamp_utc = timestamp.replace(tzinfo=None)
        else:
            timestamp_utc = timestamp

        # Prepare the watermark data
        watermark_data = {
            "run_id": f"incidents_hourly_{timestamp_utc.strftime('%Y%m%dT%H%M%S')}",
            "rows_processed": rows_processed,
            "time_trigger": timestamp_utc.strftime('%Y-%m-%dT%H:%M:%S'),
            "run_details": run_details
        }

        # Log the watermark update
        logging.info(f"Updating watermark: {watermark_data}")

        # Insert or replace the watermark row - use update_preprocessing_watermark instead
        await self.bq_client.update_preprocessing_watermark(
            preprocessed_rows=rows_processed,
            timestamp=timestamp_utc,
            run_details=run_details,
            overwrite=overwrite
        )


# For direct execution and testing
if __name__ == "__main__":
    import argparse

    # Setup more visible logging and direct console output
    print("Starting preprocessing pipeline script...")

    # Configure logging to show immediately in the console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,  # Force reconfiguration of the root logger
        handlers=[
            logging.StreamHandler()  # Add a stream handler to output to console
        ]
    )

    # Suppress OpenAI client logs about retries
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--start-date", type=str, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data")
    parser.add_argument("--overwrite-watermark", action="store_true",
                      help="Overwrite watermark records instead of appending")
    parser.add_argument("--verbose", action="store_true",
                      help="Show verbose output (quiet mode is default)")
    args = parser.parse_args()

    print(f"Arguments parsed: start_date={args.start_date}, end_date={args.end_date}")
    print(f"Overwrite={args.overwrite}, overwrite_watermark={args.overwrite_watermark}")

    try:
        # Run the pipeline with arguments
        print("Initializing preprocessing pipeline...")
        asyncio.run(PreprocessingPipeline().run(
            start_date=args.start_date,
            end_date=args.end_date,
            overwrite=args.overwrite,
            overwrite_watermark=args.overwrite_watermark,
            quiet=not args.verbose  # Quiet mode is True by default (when --verbose is not specified)
        ))
        print("Pipeline execution completed.")
    except Exception as e:
        import traceback
        print(f"ERROR: Pipeline execution failed with error: {str(e)}")
        print("Detailed error trace:")
        traceback.print_exc()
        sys.exit(1)
