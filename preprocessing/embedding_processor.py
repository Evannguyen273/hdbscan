# filepath: c:\Users\Bachn\OneDrive\Desktop\my_projects\hdbscan\preprocessing\embedding_processor.py
import logging
import time
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Set, Optional
from tqdm.auto import tqdm

class EmbeddingProcessor:
    """Enhanced embedding processing with comprehensive error handling and no fallbacks"""
    
    def __init__(self, config):
        self.config = config
        self.openai_client = self._initialize_openai_client()
        
        # Track failed incident numbers for reporting
        self.failed_incidents: Set[str] = set()
        self.failure_reasons: Dict[str, str] = {}
        
        # Track cascading failures from summarization
        self.summarization_failed_incidents: Set[str] = set()
        
        # Embedding model limits
        self.max_embedding_tokens = 8192  # for text-embedding-ada-002
        
    def _initialize_openai_client(self):
        """Initialize OpenAI client from config"""
        import openai
        return openai.AzureOpenAI(
            azure_endpoint=self.config.azure_openai.endpoint,
            api_key=self.config.azure_openai.api_key,
            api_version=self.config.azure_openai.api_version
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        if not text:
            return 0
        return len(text) // 4 + 1
    
    def validate_text_for_embedding(self, text: str, incident_number: str) -> bool:
        """
        Validate if text is suitable for embedding generation.
        
        Args:
            text: Text to validate
            incident_number: Incident number for error tracking
            
        Returns:
            True if text is valid for embedding
        """
        if not text or not text.strip():
            raise ValueError("Empty or null text - no summary available")
        
        # Check token limit
        estimated_tokens = self.estimate_tokens(text)
        if estimated_tokens > self.max_embedding_tokens:
            raise ValueError(f"Text too long for embedding ({estimated_tokens} tokens > {self.max_embedding_tokens} limit)")
        
        # Check for very short texts that might not be meaningful
        if len(text.strip()) < 10:
            raise ValueError("Text too short for meaningful embedding")
        
        return True
    
    def create_single_embedding(self, text: str, incident_number: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            incident_number: Incident number for error tracking
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            Exception: If embedding generation fails
        """
        # Validate text first
        self.validate_text_for_embedding(text, incident_number)
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text.strip(),
                timeout=30
            )
            
            embedding = np.array(response.data[0].embedding)
            
            # Validate embedding quality
            if len(embedding) == 0:
                raise ValueError("Received empty embedding from API")
            
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                raise ValueError("Received invalid embedding with NaN or Inf values")
            
            return embedding
            
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Embedding API call failed: {str(e)}")
    
    def create_batch_embeddings(self, texts: List[str], incident_numbers: List[str]) -> List[Optional[np.ndarray]]:
        """
        Create embeddings for multiple texts using batch API.
        
        Args:
            texts: List of texts to embed
            incident_numbers: Corresponding incident numbers
            
        Returns:
            List of embeddings (None for failures)
        """
        if len(texts) != len(incident_numbers):
            raise ValueError("Texts and incident numbers must have same length")
        
        try:
            # Filter out invalid texts but track their positions
            valid_indices = []
            valid_texts = []
            valid_incidents = []
            
            for i, (text, incident_num) in enumerate(zip(texts, incident_numbers)):
                try:
                    self.validate_text_for_embedding(text, incident_num)
                    valid_indices.append(i)
                    valid_texts.append(text.strip())
                    valid_incidents.append(incident_num)
                except Exception as e:
                    logging.warning("✗ %s: VALIDATION_ERROR - %s", incident_num, str(e))
                    self.failed_incidents.add(incident_num)
                    self.failure_reasons[incident_num] = str(e)
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Call batch embedding API
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=valid_texts,
                timeout=60
            )
            
            # Process response and map back to original order
            embeddings = [None] * len(texts)
            
            for i, embedding_data in enumerate(response.data):
                if i < len(valid_indices):
                    original_index = valid_indices[i]
                    embedding = np.array(embedding_data.embedding)
                    
                    # Validate embedding
                    if len(embedding) > 0 and not (np.isnan(embedding).any() or np.isinf(embedding).any()):
                        embeddings[original_index] = embedding
                        logging.debug("✓ %s: Successfully embedded", valid_incidents[i])
                    else:
                        incident_num = valid_incidents[i]
                        error_msg = "Invalid embedding received from API"
                        logging.warning("✗ %s: API_ERROR - %s", incident_num, error_msg)
                        self.failed_incidents.add(incident_num)
                        self.failure_reasons[incident_num] = error_msg
            
            return embeddings
            
        except Exception as e:
            # If batch fails, mark all as failed
            error_msg = f"Batch embedding failed: {str(e)}"
            for incident_num in incident_numbers:
                if incident_num not in self.failed_incidents:  # Don't overwrite validation errors
                    self.failed_incidents.add(incident_num)
                    self.failure_reasons[incident_num] = error_msg
                    logging.warning("✗ %s: BATCH_ERROR - %s", incident_num, error_msg)
            
            return [None] * len(texts)
    
    def process_embeddings_batch(self, summaries_series: pd.Series, batch_size: int = 50, 
                                 use_batch_api: bool = True) -> Tuple[pd.Series, Dict]:
        """
        Process summaries to create embeddings with comprehensive error handling.
        
        Args:
            summaries_series: Series with incident summaries (from summarization stage)
            batch_size: Batch size for processing
            use_batch_api: Whether to use batch API (falls back to individual if batch fails)
            
        Returns:
            Tuple of (embeddings_series, statistics)
        """
        logging.info("Starting embedding pipeline: %d summaries, batch_size=%d", len(summaries_series), batch_size)
        
        # Reset failure tracking
        self.failed_incidents.clear()
        self.failure_reasons.clear()
        self.summarization_failed_incidents.clear()
        
        # Identify incidents without summaries (failed at summarization stage)
        null_summaries = summaries_series.isnull()
        self.summarization_failed_incidents = set(summaries_series[null_summaries].index.astype(str))
        
        # Work only with valid summaries
        valid_summaries = summaries_series.dropna()
        logging.info("Valid summaries for embedding: %d/%d", len(valid_summaries), len(summaries_series))
        
        if self.summarization_failed_incidents:
            logging.warning("Skipping %d incidents due to missing summaries: %s", 
                          len(self.summarization_failed_incidents), 
                          sorted(list(self.summarization_failed_incidents))[:10])  # Show first 10
        
        # Initialize results
        embeddings_series = pd.Series(index=summaries_series.index, dtype=object)
        successful_count = 0
        total_batches = (len(valid_summaries) + batch_size - 1) // batch_size if len(valid_summaries) > 0 else 0
        
        # Process in batches
        with tqdm(total=len(valid_summaries), desc="Creating embeddings") as pbar:
            for batch_num in range(0, len(valid_summaries), batch_size):
                batch_summaries = valid_summaries.iloc[batch_num:batch_num+batch_size]
                current_batch = (batch_num // batch_size) + 1
                
                batch_start_time = time.time()
                batch_successes = 0
                batch_failures = 0
                batch_failed_incidents = []
                
                logging.info("Processing batch %d/%d - summaries %d to %d", 
                           current_batch, total_batches, batch_num+1, 
                           min(batch_num+batch_size, len(valid_summaries)))
                
                texts = batch_summaries.tolist()
                incident_numbers = [str(idx) for idx in batch_summaries.index]
                
                if use_batch_api and len(texts) > 1:
                    # Try batch API first
                    try:
                        embeddings = self.create_batch_embeddings(texts, incident_numbers)
                        
                        for i, (idx, embedding) in enumerate(zip(batch_summaries.index, embeddings)):
                            if embedding is not None:
                                embeddings_series[idx] = embedding
                                successful_count += 1
                                batch_successes += 1
                            else:
                                batch_failures += 1
                                batch_failed_incidents.append(incident_numbers[i])
                        
                    except Exception as e:
                        logging.warning("Batch API failed, falling back to individual processing: %s", str(e))
                        use_batch_api = False  # Disable batch API for remaining batches
                        
                        # Process individually as fallback
                        for idx, text in batch_summaries.items():
                            incident_number = str(idx)
                            try:
                                embedding = self.create_single_embedding(text, incident_number)
                                embeddings_series[idx] = embedding
                                successful_count += 1
                                batch_successes += 1
                                logging.debug("✓ %s: Successfully embedded (individual)", incident_number)
                            except Exception as e:
                                error_msg = str(e)
                                self.failed_incidents.add(incident_number)
                                self.failure_reasons[incident_number] = error_msg
                                batch_failures += 1
                                batch_failed_incidents.append(incident_number)
                                
                                error_type = self._classify_error(error_msg)
                                logging.warning("✗ %s: %s - %s", incident_number, error_type, error_msg)
                else:
                    # Process individually
                    for idx, text in batch_summaries.items():
                        incident_number = str(idx)
                        try:
                            embedding = self.create_single_embedding(text, incident_number)
                            embeddings_series[idx] = embedding
                            successful_count += 1
                            batch_successes += 1
                            logging.debug("✓ %s: Successfully embedded", incident_number)
                        except Exception as e:
                            error_msg = str(e)
                            self.failed_incidents.add(incident_number)
                            self.failure_reasons[incident_number] = error_msg
                            batch_failures += 1
                            batch_failed_incidents.append(incident_number)
                            
                            error_type = self._classify_error(error_msg)
                            logging.warning("✗ %s: %s - %s", incident_number, error_type, error_msg)
                
                # Update progress
                pbar.update(len(batch_summaries))
                
                # Log batch completion summary
                batch_duration = time.time() - batch_start_time
                batch_success_rate = (batch_successes / len(batch_summaries)) * 100 if len(batch_summaries) > 0 else 0
                
                logging.info("Batch %d complete: %d/%d successful (%.1f%%) in %.1fs", 
                           current_batch, batch_successes, len(batch_summaries), 
                           batch_success_rate, batch_duration)
                
                if batch_failed_incidents:
                    logging.warning("Batch %d failures: %s", current_batch, batch_failed_incidents)
                
                # Add small delay between batches to avoid rate limits
                if current_batch < total_batches:
                    time.sleep(0.3)
        
        # Create comprehensive statistics
        total_original_incidents = len(summaries_series)
        total_embedding_failures = len(self.failed_incidents)
        total_summarization_failures = len(self.summarization_failed_incidents)
        
        # Filter out failed embeddings (keep as NaN for tracking)
        embeddings_series = embeddings_series.dropna()
        
        # Create detailed failure breakdown
        failure_breakdown = {}
        for reason in ["RATE_LIMIT", "TIMEOUT", "TOKEN_LIMIT", "VALIDATION_ERROR", "BATCH_ERROR", "API_ERROR"]:
            count = len([k for k, v in self.failure_reasons.items() if reason in self._classify_error(v)])
            if count > 0:
                failure_breakdown[reason] = count
        
        stats = {
            "total_original_incidents": total_original_incidents,
            "summaries_available": len(valid_summaries),
            "successful_embeddings": successful_count,
            "summarization_failures": total_summarization_failures,
            "embedding_failures": total_embedding_failures,
            "embedding_success_rate": (successful_count / len(valid_summaries)) * 100 if len(valid_summaries) > 0 else 0,
            "overall_success_rate": (successful_count / total_original_incidents) * 100 if total_original_incidents > 0 else 0,
            "failed_incident_numbers": list(self.failed_incidents),
            "summarization_failed_incidents": list(self.summarization_failed_incidents),
            "failure_breakdown": failure_breakdown,
            "failure_reasons": dict(self.failure_reasons)
        }
        
        logging.info("Embedding complete: %d/%d successful embeddings (%.1f%% of valid summaries)", 
                    successful_count, len(valid_summaries), stats['embedding_success_rate'])
        logging.info("Overall pipeline success: %d/%d incidents have embeddings (%.1f%% of original)", 
                    successful_count, total_original_incidents, stats['overall_success_rate'])
        
        if self.failed_incidents:
            logging.warning("Embedding failures: %s", sorted(list(self.failed_incidents)))
            logging.info("Embedding failure breakdown:")
            for error_type, count in failure_breakdown.items():
                logging.info("  - %s: %d incidents", error_type, count)
        
        return embeddings_series, stats
    
    def _classify_error(self, error_msg: str) -> str:
        """Classify error types for better logging"""
        error_msg_lower = error_msg.lower()
        
        if "rate limit" in error_msg_lower or "429" in error_msg_lower:
            return "RATE_LIMIT"
        elif "timeout" in error_msg_lower:
            return "TIMEOUT"
        elif "token" in error_msg_lower or "context length" in error_msg_lower:
            return "TOKEN_LIMIT"
        elif "empty" in error_msg_lower or "null" in error_msg_lower or "short" in error_msg_lower:
            return "VALIDATION_ERROR"
        elif "batch" in error_msg_lower:
            return "BATCH_ERROR"
        elif "authentication" in error_msg_lower or "401" in error_msg_lower:
            return "AUTH_ERROR"
        elif "connection" in error_msg_lower:
            return "CONNECTION_ERROR"
        elif "service unavailable" in error_msg_lower or "503" in error_msg_lower:
            return "SERVICE_UNAVAILABLE"
        else:
            return "API_ERROR"
    
    def get_clustering_ready_data(self, embeddings_series: pd.Series) -> Tuple[np.ndarray, pd.Index]:
        """
        Prepare embeddings data for clustering by converting to matrix format.
        
        Args:
            embeddings_series: Series with embeddings
            
        Returns:
            Tuple of (embedding_matrix, valid_indices)
        """
        valid_embeddings = embeddings_series.dropna()
        
        if len(valid_embeddings) == 0:
            logging.warning("No valid embeddings available for clustering")
            return np.array([]), pd.Index([])
        
        # Convert to matrix
        embedding_matrix = np.vstack(valid_embeddings.values)
        
        logging.info("Clustering data prepared: %d embeddings with %d dimensions", 
                    embedding_matrix.shape[0], embedding_matrix.shape[1])
        
        return embedding_matrix, valid_embeddings.index
    
    def get_failed_incidents_report(self) -> Dict:
        """Get comprehensive report of failed incidents for troubleshooting"""
        return {
            "embedding_failures": {
                "count": len(self.failed_incidents),
                "incident_numbers": sorted(list(self.failed_incidents)),
                "failure_breakdown": {
                    reason: len([k for k, v in self.failure_reasons.items() if reason in self._classify_error(v)])
                    for reason in ["RATE_LIMIT", "TIMEOUT", "TOKEN_LIMIT", "VALIDATION_ERROR", "BATCH_ERROR", "API_ERROR"]
                },
                "detailed_failures": dict(self.failure_reasons)
            },
            "summarization_failures": {
                "count": len(self.summarization_failed_incidents),
                "incident_numbers": sorted(list(self.summarization_failed_incidents))
            },
            "total_pipeline_failures": len(self.failed_incidents) + len(self.summarization_failed_incidents)
        }