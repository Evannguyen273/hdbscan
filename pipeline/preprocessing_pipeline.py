# pipeline/preprocessing_pipeline.py
# Preprocessing pipeline for cumulative training approach
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import asyncio

from config.config import get_config
from preprocessing.orchestrator import PreprocessingOrchestrator

class PreprocessingPipeline:
    """
    Preprocessing pipeline for cumulative HDBSCAN training.
    Handles text processing, embedding generation, and data preparation.
    """
    
    def __init__(self, config=None):
        """Initialize preprocessing pipeline with updated config system"""
        self.config = config if config is not None else get_config()
        
        # Initialize preprocessing orchestrator
        self.orchestrator = PreprocessingOrchestrator(self.config)
        
        # Processing statistics
        self.processing_stats = {}
        
        logging.info("Preprocessing pipeline initialized for cumulative training")
    
    async def process_for_training(self, dataset: pd.DataFrame) -> Dict[str, Dict]:
        """
        Process dataset for training across all tech centers.
        
        Args:
            dataset: Raw incident dataset with cumulative data
            
        Returns:
            Dictionary with preprocessing results by tech center
        """
        processing_start = datetime.now()
        
        logging.info("Starting preprocessing for training with %d incidents", len(dataset))
        
        try:
            # Group data by tech center
            tech_center_data = self._group_by_tech_center(dataset)
            
            # Process each tech center in parallel
            processing_results = await self._process_tech_centers_parallel(tech_center_data)
            
            processing_duration = datetime.now() - processing_start
            
            # Compile overall statistics
            total_incidents = len(dataset)
            successful_incidents = sum(
                result.get('incidents_processed', 0) 
                for result in processing_results.values()
                if result.get('status') == 'success'
            )
            
            overall_stats = {
                "processing_duration_seconds": processing_duration.total_seconds(),
                "total_incidents": total_incidents,
                "successful_incidents": successful_incidents,
                "success_rate": (successful_incidents / total_incidents * 100) if total_incidents > 0 else 0,
                "tech_centers_processed": len([r for r in processing_results.values() if r.get('status') == 'success']),
                "tech_centers_failed": len([r for r in processing_results.values() if r.get('status') == 'failed'])
            }
            
            self.processing_stats = overall_stats
            
            logging.info("Preprocessing completed: %.1f%% success rate (%d/%d incidents)",
                        overall_stats['success_rate'], successful_incidents, total_incidents)
            
            return processing_results
            
        except Exception as e:
            logging.error("Preprocessing for training failed: %s", str(e))
            raise
    
    async def process_for_prediction(self, incidents: pd.DataFrame) -> Dict:
        """
        Process incidents for real-time prediction.
        
        Args:
            incidents: Raw incident data for prediction
            
        Returns:
            Preprocessed data ready for prediction
        """
        processing_start = datetime.now()
        
        logging.info("Starting preprocessing for prediction with %d incidents", len(incidents))
        
        try:
            # Get batch size from config
            batch_size = self.config.prediction.batch_size
            
            # Process in smaller batches for prediction
            if len(incidents) > batch_size:
                logging.warning("Large prediction batch (%d), processing in chunks", len(incidents))
                
                results = []
                for i in range(0, len(incidents), batch_size):
                    batch = incidents.iloc[i:i+batch_size]
                    batch_result = await self._process_prediction_batch(batch)
                    results.append(batch_result)
                
                # Combine results
                combined_result = self._combine_batch_results(results)
            else:
                combined_result = await self._process_prediction_batch(incidents)
            
            processing_duration = datetime.now() - processing_start
            combined_result['processing_duration_seconds'] = processing_duration.total_seconds()
            
            logging.info("Prediction preprocessing completed in %.2f seconds",
                        processing_duration.total_seconds())
            
            return combined_result
            
        except Exception as e:
            logging.error("Preprocessing for prediction failed: %s", str(e))
            raise
      def _group_by_tech_center(self, dataset: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group dataset by tech center or create single group if no tech_center column"""
        tech_center_data = {}
        
        # Check if tech_center column exists
        if 'tech_center' in dataset.columns:
            # Group by existing tech_center column
            for tech_center in self.config.tech_centers:
                tc_data = dataset[dataset['tech_center'] == tech_center].copy()
                if len(tc_data) > 0:
                    tech_center_data[tech_center] = tc_data
                    logging.info("Tech center %s: %d incidents", tech_center, len(tc_data))
                else:
                    logging.warning("No data found for tech center: %s", tech_center)
        else:
            # No tech_center column - treat all data as one group
            logging.info("No tech_center column found, processing all data as 'All_Centers'")
            tech_center_data['All_Centers'] = dataset.copy()
            logging.info("All Centers: %d incidents", len(dataset))
        
        return tech_center_data
    
    async def _process_tech_centers_parallel(self, tech_center_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Process tech centers in parallel"""
        max_workers = self.config.training.processing.get('max_workers', 4)
        
        # Create processing tasks
        tasks = []
        for tech_center, data in tech_center_data.items():
            task = self._process_single_tech_center(tech_center, data)
            tasks.append((tech_center, task))
        
        # Run with concurrency limit
        results = {}
        semaphore = asyncio.Semaphore(max_workers)
        
        async def bounded_processing(tech_center: str, task):
            async with semaphore:
                return await task
        
        # Execute all tasks
        for tech_center, task in tasks:
            try:
                result = await bounded_processing(tech_center, task)
                results[tech_center] = result
                logging.info("Processing completed for %s", tech_center)
            except Exception as e:
                logging.error("Processing failed for %s: %s", tech_center, str(e))
                results[tech_center] = {
                    "status": "failed",
                    "error": str(e),
                    "incidents_processed": 0
                }
        
        return results
    
    async def _process_single_tech_center(self, tech_center: str, data: pd.DataFrame) -> Dict:
        """Process data for a single tech center"""
        processing_start = datetime.now()
        
        try:
            logging.info("Processing %s with %d incidents", tech_center, len(data))
            
            # Check minimum incident requirement
            min_incidents = self.config.clustering.min_incidents_per_domain
            if len(data) < min_incidents:
                logging.warning("Insufficient incidents for %s (%d < %d)", 
                               tech_center, len(data), min_incidents)
                return {
                    "status": "failed",
                    "reason": "insufficient_incidents",
                    "incidents_processed": 0,
                    "required_minimum": min_incidents
                }
            
            # Apply incident limit if configured
            max_incidents = self.config.training.processing.get('max_incidents_per_training', 100000)
            if len(data) > max_incidents:
                logging.info("Limiting %s incidents from %d to %d", tech_center, len(data), max_incidents)
                data = data.sample(n=max_incidents, random_state=42).reset_index(drop=True)            # Run preprocessing pipeline using the new incident processing method
            azure_config = self.config.azure.config
            batch_size_summarization = azure_config.get('summarization_batch_size', 10)
            
            # Use the new text processor method for incident dataframes
            from preprocessing.text_processing import TextProcessor
            text_processor = TextProcessor(self.config)
            
            # Process incidents to create combined summaries
            summaries_series, text_stats = await text_processor.process_incident_for_embedding_batch(
                data, batch_size=batch_size_summarization
            )
            
            # Generate embeddings from summaries
            from preprocessing.embedding_generation import EmbeddingGenerator
            embedding_generator = EmbeddingGenerator(self.config)
            
            # Filter out failed summaries (NaN values)
            valid_summaries = summaries_series.dropna()
            valid_indices = valid_summaries.index
            
            if len(valid_summaries) == 0:
                result = {
                    "status": "failed",
                    "reason": "no_valid_summaries",
                    "incidents_processed": 0,
                    "total_input_incidents": len(data),
                    "text_processing_stats": text_stats
                }
                return result
            
            # Generate embeddings for valid summaries
            embeddings_array, embedding_stats = await embedding_generator.generate_embeddings_batch(
                valid_summaries.tolist(),
                batch_size=azure_config.get('embedding_batch_size', 50)
            )
            
            if len(embeddings_array) > 0:
                result = {
                    "status": "success",
                    "tech_center": tech_center,
                    "embeddings": embeddings_array,
                    "incident_data": data.iloc[valid_indices],
                    "summaries": valid_summaries,
                    "incidents_processed": len(valid_indices),
                    "total_input_incidents": len(data),
                    "processing_duration_seconds": processing_duration.total_seconds(),
                    "text_processing_stats": text_stats,
                    "embedding_stats": embedding_stats
                }
            else:
                result = {
                    "status": "failed",
                    "reason": "no_valid_embeddings",
                    "incidents_processed": 0,
                    "total_input_incidents": len(data),
                    "text_processing_stats": text_stats,
                    "embedding_stats": embedding_stats
                }
            
            return result
            
        except Exception as e:
            processing_duration = datetime.now() - processing_start
            logging.error("Processing failed for %s: %s", tech_center, str(e))
            return {
                "status": "failed",
                "error": str(e),
                "incidents_processed": 0,
                "processing_duration_seconds": processing_duration.total_seconds()
            }
    
    async def _process_prediction_batch(self, incidents: pd.DataFrame) -> Dict:
        """Process a batch of incidents for prediction"""
        try:
            # Simplified preprocessing for prediction
            # In real implementation, this would be similar to training preprocessing
            # but optimized for speed
            
            summaries, embeddings, valid_indices, stats = self.orchestrator.run_complete_pipeline(
                incidents,
                summarization_batch_size=5,  # Smaller batches for faster prediction
                embedding_batch_size=25,
                use_batch_embedding_api=True
            )
            
            if len(embeddings) > 0:
                return {
                    "status": "success",
                    "embeddings": embeddings,
                    "incident_data": incidents.iloc[valid_indices],
                    "summaries": summaries,
                    "incidents_processed": len(valid_indices),
                    "preprocessing_stats": stats
                }
            else:
                return {
                    "status": "failed",
                    "reason": "no_valid_embeddings",
                    "incidents_processed": 0,
                    "preprocessing_stats": stats
                }
                
        except Exception as e:
            logging.error("Prediction batch processing failed: %s", str(e))
            return {
                "status": "failed",
                "error": str(e),
                "incidents_processed": 0
            }
    
    def _combine_batch_results(self, batch_results: List[Dict]) -> Dict:
        """Combine results from multiple prediction batches"""
        combined = {
            "status": "success",
            "embeddings": [],
            "incident_data": pd.DataFrame(),
            "summaries": pd.Series(),
            "incidents_processed": 0,
            "total_batches": len(batch_results)
        }
        
        for batch_result in batch_results:
            if batch_result.get("status") == "success":
                combined["embeddings"].extend(batch_result["embeddings"])
                combined["incident_data"] = pd.concat([
                    combined["incident_data"], 
                    batch_result["incident_data"]
                ], ignore_index=True)
                combined["summaries"] = pd.concat([
                    combined["summaries"],
                    batch_result["summaries"]
                ], ignore_index=True)
                combined["incidents_processed"] += batch_result["incidents_processed"]
        
        # Convert embeddings list to numpy array
        if combined["embeddings"]:
            combined["embeddings"] = np.vstack(combined["embeddings"])
        
        return combined
    
    def validate_preprocessing_config(self) -> Dict:
        """Validate preprocessing configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
          # Check Azure configuration
        azure_config = self.config.azure.openai
        if not azure_config.get('endpoint'):
            validation_results["errors"].append("Azure OpenAI endpoint not configured")
            validation_results["valid"] = False
        
        if not azure_config.get('api_key'):
            validation_results["errors"].append("Azure OpenAI API key not configured")
            validation_results["valid"] = False
        
        # Check batch sizes
        training_config = self.config.training.processing
        max_incidents = training_config.get('max_incidents_per_training', 100000)
        if max_incidents > 200000:
            validation_results["warnings"].append(
                f"Large max_incidents_per_training setting ({max_incidents}) may cause memory issues"
            )
        
        # Check tech centers
        if len(self.config.tech_centers) == 0:
            validation_results["errors"].append("No tech centers configured")
            validation_results["valid"] = False
        
        return validation_results
    
    def get_processing_statistics(self) -> Dict:
        """Get preprocessing pipeline statistics"""
        return self.processing_stats
    
    async def cleanup_resources(self):
        """Cleanup any resources used by preprocessing"""
        # Clear any cached data or connections
        self.processing_stats = {}
        logging.info("Preprocessing pipeline resources cleaned up")
    
    async def llm_summarization(self, incidents: pd.DataFrame) -> pd.Series:
        """
        Perform LLM-based summarization on incident data.
        
        Args:
            incidents: DataFrame containing incident data for summarization
            
        Returns:
            Series with summarized text for each incident
        """
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential
        
        # Initialize Text Analytics client
        ta_client = TextAnalyticsClient(
            endpoint=self.config.azure.openai.endpoint,
            credential=AzureKeyCredential(self.config.azure.openai.api_key)
        )
        
        # Prepare documents for summarization
        documents = incidents['incident_description'].tolist()
        
        try:
            # Call the Text Analytics API for summarization
            response = await ta_client.summarize(documents=documents)
            
            # Extract and return summaries
            summaries = [doc.summary for doc in response if not doc.is_error]
            
            # Handle any errors in the response
            for doc in response:
                if doc.is_error:
                    logging.error("Error summarizing document: %s", doc.error)
            
            return pd.Series(summaries)
        
        except Exception as e:
            logging.error("LLM summarization failed: %s", str(e))
            raise