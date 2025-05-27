import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import os

from data.bigquery_client import BigQueryClient
from data.blob_storage import BlobStorageClient
from preprocessing.text_processing import TextProcessor
from preprocessing.embedding_generation import EmbeddingGenerator

class PreprocessingPipeline:
    """
    Separate preprocessing pipeline that runs every hour to:
    1. Detect new incidents using sys_created_on watermark
    2. Generate combined_incidents_summary 
    3. Create embeddings
    4. Store in preprocessing table for training/prediction
    """
    
    def __init__(self, config):
        self.config = config
        self.bigquery_client = BigQueryClient(config)
        self.blob_storage = BlobStorageClient(config)
        self.text_processor = TextProcessor(config)
        self.embedding_generator = EmbeddingGenerator(config)
        
        # Table names
        self.incidents_table = f"{config.bigquery.project_id}.{config.bigquery.datasets.embeddings}.{config.bigquery.tables.incidents}"
        self.preprocessed_table = f"{config.bigquery.project_id}.{config.bigquery.datasets.preprocessing}.{config.bigquery.tables.preprocessed}"
        
    def get_watermark_timestamp(self, tech_center: str) -> Optional[datetime]:
        """Get the last processed timestamp for a tech center"""
        try:
            query = f"""
            SELECT MAX(sys_created_on) as last_processed
            FROM `{self.preprocessed_table}`
            WHERE tech_center = '{tech_center}'
            """
            result = self.bigquery_client.run_query(query)
            if not result.empty and result.iloc[0]['last_processed'] is not None:
                return result.iloc[0]['last_processed']
            return None
        except Exception as e:
            logging.warning(f"Could not get watermark for {tech_center}: {e}")
            return None
    
    def update_watermark_timestamp(self, tech_center: str, timestamp: datetime):
        """Update the watermark timestamp for a tech center"""
        try:
            # Store watermark in a separate tracking table
            watermark_table = f"{self.config.bigquery.project_id}.{self.config.bigquery.datasets.preprocessing}.preprocessing_watermarks"
            
            query = f"""
            MERGE `{watermark_table}` T
            USING (SELECT '{tech_center}' as tech_center, '{timestamp}' as last_processed_timestamp) S
            ON T.tech_center = S.tech_center
            WHEN MATCHED THEN
                UPDATE SET last_processed_timestamp = S.last_processed_timestamp
            WHEN NOT MATCHED THEN
                INSERT (tech_center, last_processed_timestamp, updated_at)
                VALUES (S.tech_center, S.last_processed_timestamp, CURRENT_TIMESTAMP())
            """
            self.bigquery_client.run_query(query)
        except Exception as e:
            logging.error(f"Failed to update watermark for {tech_center}: {e}")
    
    def get_new_incidents(self, tech_center: str, watermark: Optional[datetime] = None) -> pd.DataFrame:
        """Get new incidents for a tech center since the watermark"""
        
        # If no watermark, get last 24 hours
        if watermark is None:
            watermark = datetime.now() - timedelta(days=1)
        
        query = f"""
        SELECT 
            number,
            sys_created_on,
            business_service,
            short_description,
            description,
            tech_center
        FROM `{self.incidents_table}`
        WHERE tech_center = '{tech_center}'
            AND sys_created_on > '{watermark}'
            AND sys_created_on <= CURRENT_TIMESTAMP()
        ORDER BY sys_created_on
        """
        
        logging.info(f"Getting new incidents for {tech_center} since {watermark}")
        df = self.bigquery_client.run_query(query)
        logging.info(f"Found {len(df)} new incidents for {tech_center}")
        
        return df
    
    def process_incidents_batch(self, df: pd.DataFrame, tech_center: str) -> pd.DataFrame:
        """Process a batch of incidents to generate summaries and embeddings"""
        
        if df.empty:
            return df
        
        logging.info(f"Processing {len(df)} incidents for {tech_center}")
        
        # Generate combined summaries
        combined_summaries, fallback_stats = self.text_processor.process_incident_for_embedding_batch(
            df, batch_size=self.config.pipeline.preprocessing.batch_size
        )
        df['combined_incidents_summary'] = combined_summaries
        
        # Generate embeddings (only semantic as per config)
        df_with_embeddings, classification_result, embedding_stats = self.embedding_generator.create_hybrid_embeddings(
            df, text_column='combined_incidents_summary'
        )
        
        # Select only required columns for storage
        processed_df = df_with_embeddings[[
            'number', 
            'sys_created_on',
            'combined_incidents_summary',
            'embedding',
            'tech_center'
        ]].copy()
        
        # Add processing metadata
        processed_df['processed_at'] = datetime.now()
        processed_df['processing_version'] = '1.0'
        
        return processed_df
    
    def save_preprocessed_data(self, df: pd.DataFrame, tech_center: str):
        """Save preprocessed data to BigQuery table"""
        
        if df.empty:
            return
        
        try:
            success = self.bigquery_client.save_dataframe(
                df, 
                self.preprocessed_table, 
                write_disposition="WRITE_APPEND"
            )
            
            if success:
                logging.info(f"Successfully saved {len(df)} preprocessed incidents for {tech_center}")
                
                # Update watermark with the latest sys_created_on
                latest_timestamp = df['sys_created_on'].max()
                self.update_watermark_timestamp(tech_center, latest_timestamp)
            else:
                logging.error(f"Failed to save preprocessed data for {tech_center}")
                
        except Exception as e:
            logging.error(f"Error saving preprocessed data for {tech_center}: {e}")
    
    def run_preprocessing_for_tech_center(self, tech_center: str) -> Dict:
        """Run preprocessing for a single tech center"""
        
        start_time = time.time()
        logging.info(f"Starting preprocessing for {tech_center}")
        
        try:
            # Get watermark
            watermark = self.get_watermark_timestamp(tech_center)
            
            # Get new incidents
            new_incidents = self.get_new_incidents(tech_center, watermark)
            
            if new_incidents.empty:
                logging.info(f"No new incidents found for {tech_center}")
                return {
                    "tech_center": tech_center,
                    "status": "success",
                    "processed_count": 0,
                    "runtime_seconds": time.time() - start_time
                }
            
            # Process incidents
            processed_data = self.process_incidents_batch(new_incidents, tech_center)
            
            # Save to BigQuery
            self.save_preprocessed_data(processed_data, tech_center)
            
            runtime = time.time() - start_time
            logging.info(f"Preprocessing completed for {tech_center} in {runtime:.2f} seconds")
            
            return {
                "tech_center": tech_center,
                "status": "success", 
                "processed_count": len(processed_data),
                "runtime_seconds": runtime
            }
            
        except Exception as e:
            logging.error(f"Preprocessing failed for {tech_center}: {e}")
            return {
                "tech_center": tech_center,
                "status": "failed",
                "error": str(e),
                "runtime_seconds": time.time() - start_time
            }
    
    def run_preprocessing_all_tech_centers(self) -> Dict:
        """Run preprocessing for all tech centers"""
        
        start_time = time.time()
        logging.info("Starting preprocessing for all tech centers")
        
        results = []
        total_processed = 0
        
        for tech_center in self.config.pipeline.tech_centers:
            result = self.run_preprocessing_for_tech_center(tech_center)
            results.append(result)
            total_processed += result.get("processed_count", 0)
        
        # Summary
        successful = len([r for r in results if r["status"] == "success"])
        failed = len([r for r in results if r["status"] == "failed"])
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tech_centers": len(self.config.pipeline.tech_centers),
            "successful": successful,
            "failed": failed,
            "total_processed": total_processed,
            "runtime_seconds": time.time() - start_time,
            "results": results
        }
        
        logging.info(f"Preprocessing completed: {successful} successful, {failed} failed, {total_processed} total processed")
        
        # Save results to blob storage for monitoring
        if self.config.pipeline.save_to_local:
            output_dir = f"{self.config.pipeline.result_path}/preprocessing/logs"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"{output_dir}/preprocessing_run_{timestamp}.json", "w") as f:
                import json
                json.dump(summary, f, indent=2)
        
        return summary