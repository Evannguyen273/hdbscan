# preprocessing/embedding_generation.py
# Updated for new config structure and cumulative training approach
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import openai
from openai import AsyncAzureOpenAI
import time
from datetime import datetime

from config.config import get_config

class EmbeddingGenerator:
    """
    Embedding generator for Azure OpenAI with batch processing support.
    Updated for cumulative training approach.
    """
    
    def __init__(self, config=None):
        """Initialize embedding generator with updated config system"""
        self.config = config if config is not None else get_config()
        
        # Initialize Azure OpenAI client
        self._init_azure_client()
        
        # Processing statistics
        self.generation_stats = {
            "total_embeddings_generated": 0,
            "total_api_calls": 0,
            "total_processing_time": 0,
            "batch_successes": 0,
            "batch_failures": 0
        }
        
        # Validation statistics
        self.validation_stats = {
            'total_processed': 0,
            'valid_texts': 0,
            'empty_texts': 0,
            'too_long_texts': 0,
            'too_short_texts': 0,
            'api_failures': 0
        }
        
        # Incident tracking
        self.failed_incidents = {}
        self.failure_reasons = {}
        
        logging.info("Embedding generator initialized with Azure OpenAI")
    
    def _init_azure_client(self):
        """Initialize Azure OpenAI client"""
        azure_config = self.config.azure.openai
        
        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_config.get('endpoint'),
            api_key=azure_config.get('api_key'),
            api_version=azure_config.get('api_version', '2024-02-01')
        )
        
        self.embedding_model = azure_config.get('embedding_model', 'text-embedding-ada-002')
        self.max_retries = azure_config.get('max_retries', 3)
        self.retry_delay = azure_config.get('retry_delay_seconds', 1)
    
    def _validate_and_prepare_text(self, text: str, index: int) -> Optional[str]:
        """
        Validate and prepare text for embedding generation with comprehensive checks
        """
        self.validation_stats['total_processed'] += 1
        
        # Check if text is string and not empty
        if not isinstance(text, str) or not text.strip():
            self.failed_incidents[index] = "empty_or_invalid"
            self.failure_reasons[index] = "Empty or non-string text"
            self.validation_stats['empty_texts'] += 1
            return None
        
        text = text.strip()
        
        # Check minimum length for meaningful embedding
        if len(text) < self.min_text_length:
            self.failed_incidents[index] = "too_short"
            self.failure_reasons[index] = f"Text too short ({len(text)} chars < {self.min_text_length} minimum)"
            self.validation_stats['too_short_texts'] += 1
            return None
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4
        if estimated_tokens == 0 and len(text) > 0:
            estimated_tokens = 1
        
        # Handle texts that exceed token limit
        if estimated_tokens > self.max_embedding_tokens:
            logging.warning(
                f"Text at index {index} too long ({estimated_tokens} tokens > {self.max_embedding_tokens} limit). Truncating."
            )
            # Truncate based on approximate character count to fit token limit
            max_chars = self.max_embedding_tokens * 4
            text = text[:max_chars].strip()
            self.validation_stats['too_long_texts'] += 1
        
        self.validation_stats['valid_texts'] += 1
        return text

    async def generate_embeddings_batch(self, texts: List[str], 
                                      batch_size: int = 50,
                                      use_batch_api: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Generate embeddings for a batch of texts with enhanced validation and error handling
        """
        generation_start = datetime.now()
        
        logging.info("Generating embeddings for %d texts (batch_size=%d)", 
                    len(texts), batch_size)
        
        # Reset validation stats for this batch
        self.validation_stats = {
            'total_processed': 0,
            'valid_texts': 0,
            'empty_texts': 0,
            'too_long_texts': 0,
            'too_short_texts': 0,
            'api_failures': 0
        }
        
        try:
            # Validate and prepare texts
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                validated_text = self._validate_and_prepare_text(text, i)
                if validated_text is not None:
                    valid_texts.append(validated_text)
                    valid_indices.append(i)
            
            logging.info("Validated %d texts: %d valid, %d failed (%d empty, %d too_short, %d too_long)", 
                        len(texts), len(valid_texts), len(self.failed_incidents),
                        self.validation_stats['empty_texts'],
                        self.validation_stats['too_short_texts'], 
                        self.validation_stats['too_long_texts'])
            
            if len(valid_texts) == 0:
                logging.warning("No valid texts found for embedding generation")
                return np.array([]), []
            
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i+batch_size]
                batch_indices = valid_indices[i:i+batch_size]
                
                try:
                    batch_embeddings = await self._generate_batch_with_retry(batch_texts, i)
                    
                    if len(batch_embeddings) == len(batch_texts):
                        all_embeddings.extend(batch_embeddings)
                        logging.debug("Successfully processed batch %d (%d embeddings)", 
                                    i // batch_size, len(batch_embeddings))
                    else:
                        logging.warning("Batch %d: Expected %d embeddings, got %d", 
                                      i // batch_size, len(batch_texts), len(batch_embeddings))
                        # Still add partial results
                        all_embeddings.extend(batch_embeddings)
                
                except Exception as e:
                    logging.error("Batch %d failed completely: %s", i // batch_size, str(e))
                    self.validation_stats['api_failures'] += len(batch_texts)
                    # Track failed incidents
                    for idx in batch_indices:
                        self.failed_incidents[idx] = "api_failure"
                        self.failure_reasons[idx] = f"API failure: {str(e)}"
                    continue
            
            # Convert to numpy array and return results
            if all_embeddings:
                embeddings_array = np.array(all_embeddings)
                
                generation_duration = datetime.now() - generation_start
                logging.info("Generated %d embeddings in %.2f seconds (%.2f embeddings/sec)", 
                           len(embeddings_array), generation_duration.total_seconds(),
                           len(embeddings_array) / generation_duration.total_seconds())
                
                return embeddings_array, valid_indices[:len(all_embeddings)]
            else:
                logging.error("No embeddings generated successfully")
                return np.array([]), []
                
        except Exception as e:
            logging.error("Embedding generation failed: %s", str(e))
            return np.array([]), []

    async def _generate_batch_with_retry(self, texts: List[str], batch_start_idx: int) -> List[List[float]]:
        """Generate embeddings for a batch with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=self.embedding_model
                )
                
                embeddings = [data.embedding for data in response.data]
                return embeddings
                
            except Exception as e:
                wait_time = self.retry_delay * (2 ** attempt)
                logging.warning("Batch %d attempt %d failed: %s. Waiting %.1f seconds", 
                              batch_start_idx // 50, attempt + 1, str(e), wait_time)
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        return []

    async def _generate_batch_embeddings(self, texts: List[str], 
                                       batch_start_idx: int) -> List[List[float]]:
        """Generate embeddings for a single batch"""
        for attempt in range(self.max_retries):
            try:
                logging.debug("Generating embeddings for batch starting at %d (attempt %d)", 
                            batch_start_idx, attempt + 1)
                
                # Call Azure OpenAI API
                response = await self.client.embeddings.create(
                    input=texts,
                    model=self.embedding_model
                )
                
                # Extract embeddings from response
                embeddings = []
                for embedding_data in response.data:
                    embeddings.append(embedding_data.embedding)
                
                self.generation_stats["total_api_calls"] += 1
                
                logging.debug("Successfully generated %d embeddings for batch %d", 
                            len(embeddings), batch_start_idx)
                
                return embeddings
                
            except openai.RateLimitError as e:
                wait_time = self.retry_delay * (2 ** attempt)
                logging.warning("Rate limit hit for batch %d, waiting %.1f seconds", 
                              batch_start_idx, wait_time)
                await asyncio.sleep(wait_time)
                
            except openai.APIError as e:
                logging.error("API error for batch %d: %s", batch_start_idx, str(e))
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
                    
            except Exception as e:
                logging.error("Unexpected error for batch %d: %s", batch_start_idx, str(e))
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
        
        # If all retries failed
        logging.error("All retry attempts failed for batch %d", batch_start_idx)
        return []
    
    def _filter_valid_texts(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """Filter out empty or invalid texts"""
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                # Truncate if too long (Azure OpenAI has token limits)
                if len(text) > 8000:  # Rough character limit
                    text = text[:8000] + "..."
                
                valid_texts.append(text.strip())
                valid_indices.append(i)
            else:
                logging.debug("Skipping invalid text at index %d", i)
        
        logging.info("Filtered %d valid texts from %d total", 
                    len(valid_texts), len(texts))
        
        return valid_texts, valid_indices
    
    async def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        try:
            embeddings, _ = await self.generate_embeddings_batch([text], batch_size=1)
            
            if len(embeddings) > 0:
                return embeddings[0].tolist()
            else:
                return None
                
        except Exception as e:
            logging.error("Single embedding generation failed: %s", str(e))
            return None
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        stats = self.generation_stats.copy()
        
        # Calculate derived metrics
        if stats["batch_successes"] + stats["batch_failures"] > 0:
            stats["batch_success_rate"] = (
                stats["batch_successes"] / 
                (stats["batch_successes"] + stats["batch_failures"]) * 100
            )
        else:
            stats["batch_success_rate"] = 0
        
        if stats["total_api_calls"] > 0:
            stats["avg_embeddings_per_call"] = (
                stats["total_embeddings_generated"] / stats["total_api_calls"]
            )
        else:
            stats["avg_embeddings_per_call"] = 0
        
        if stats["total_processing_time"] > 0:
            stats["embeddings_per_second"] = (
                stats["total_embeddings_generated"] / stats["total_processing_time"]
            )
        else:
            stats["embeddings_per_second"] = 0
        
        return stats
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate embedding generation configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        azure_config = self.config.azure.openai
        
        # Check required configuration
        if not azure_config.get('endpoint'):
            validation_results["errors"].append("Azure OpenAI endpoint not configured")
            validation_results["valid"] = False
        
        if not azure_config.get('api_key'):
            validation_results["errors"].append("Azure OpenAI API key not configured")
            validation_results["valid"] = False
        
        if not azure_config.get('embedding_model'):
            validation_results["warnings"].append("Embedding model not specified, using default")
        
        # Check model availability (this would require an API call in practice)
        embedding_model = azure_config.get('embedding_model', 'text-embedding-ada-002')
        if embedding_model not in ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']:
            validation_results["warnings"].append(f"Unknown embedding model: {embedding_model}")
        
        return validation_results
    
    def reset_statistics(self):
        """Reset generation statistics"""
        self.generation_stats = {
            "total_embeddings_generated": 0,
            "total_api_calls": 0,
            "total_processing_time": 0,
            "batch_successes": 0,
            "batch_failures": 0
        }
        logging.info("Embedding generation statistics reset")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation and processing report"""
        return {
            'validation_stats': self.validation_stats.copy(),
            'failed_incidents': dict(self.failed_incidents),
            'failure_reasons': dict(self.failure_reasons),
            'success_rate': (
                self.validation_stats['valid_texts'] / 
                max(1, self.validation_stats['total_processed']) * 100
            )
        }