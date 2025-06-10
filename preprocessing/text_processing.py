# preprocessing/text_processing.py
# Updated for new config structure and cumulative training approach
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
import asyncio
from datetime import datetime
import openai
from openai import AsyncAzureOpenAI, AzureOpenAI

from config.config import get_config

class TextProcessor:
    """
    Text processor for incident data preprocessing and summarization.
    """

    def __init__(self, config):
        """Initialize TextProcessor with configuration"""
        self.config = config

        # Extract configuration values with defaults
        summarization_config = getattr(self.config.preprocessing, 'summarization', {})
        self.max_retries = summarization_config.get('max_retries', 3)
        self.batch_size = summarization_config.get('batch_size', 5)
        self.retry_delay = 1.0  # Add default retry delay of 1 second

        # Initialize text processing configuration
        self.text_config = self._get_text_processing_config()

        # Initialize statistics dictionary
        self.processing_stats = {
            "texts_processed": 0,
            "texts_cleaned": 0,
            "texts_summarized": 0,
            "summarization_failures": 0,
            "total_processing_time": 0
        }

        # Initialize OpenAI client if summarization is enabled
        self.client = None
        if summarization_config.get('enabled', True):
            self._init_openai_client()

    def _init_openai_client(self):
        """Initialize Azure OpenAI client for summarization"""
        try:
            from openai import AsyncAzureOpenAI

            # Get Azure OpenAI configuration
            azure_openai_config = self.config.azure.openai

            # Initialize client only once
            self.client = AsyncAzureOpenAI(
                api_key=azure_openai_config.api_key,
                api_version=azure_openai_config.api_version,
                azure_endpoint=azure_openai_config.endpoint
            )
            logging.info("Azure OpenAI client initialized successfully")
        except ImportError:
            logging.warning("OpenAI package not installed - summarization will be disabled")
        except Exception as e:
            logging.error(f"Failed to initialize Azure OpenAI client: {str(e)}")

    def _get_text_processing_config(self) -> Dict[str, Any]:
        """Get text processing configuration with defaults"""
        # Get preprocessing configuration
        clean_text_config = getattr(self.config.preprocessing, 'clean_text', {})

        # Provide defaults for missing values
        return {
            "enable_summarization": getattr(self.config.preprocessing.summarization, 'enabled', True),
            "summarization_batch_size": getattr(self.config.preprocessing.summarization, 'batch_size', 10),
            "max_input_length": getattr(self.config.preprocessing.summarization, 'max_input_chars', 4000),
            "max_summary_length": getattr(self.config.preprocessing.summarization, 'max_summary_length', 200),
            "clean_text": True,
            "remove_urls": clean_text_config.get('remove_urls', True),
            "remove_emails": clean_text_config.get('remove_emails', True),
            "normalize_whitespace": clean_text_config.get('normalize_whitespace', True),
            "min_text_length": clean_text_config.get('min_text_length', 10)
        }

    async def process_texts_for_training(self, texts: List[str],
                                       batch_size: int = 10) -> Tuple[np.ndarray, List[int], Dict]:
        """Process texts for training using batching and Azure OpenAI"""
        processing_start = datetime.now()

        logging.info("Processing %d texts for training", len(texts))

        try:
            # Stage 1: Clean and filter texts
            cleaned_texts, valid_indices = self._clean_and_filter_texts(texts)

            if len(cleaned_texts) == 0:
                logging.warning("No valid texts after cleaning")
                return pd.Series([]), [], {"status": "failed", "reason": "no_valid_texts"}

            # Stage 2: Summarize texts if enabled
            if self.text_config["enable_summarization"]:
                summarized_texts = await self._summarize_texts_batch(cleaned_texts, batch_size)
                processed_texts = pd.Series(summarized_texts, name='processed_text')
            else:
                processed_texts = pd.Series(cleaned_texts, name='processed_text')

            # Update statistics
            processing_duration = datetime.now() - processing_start
            self.processing_stats["texts_processed"] += len(processed_texts)
            self.processing_stats["total_processing_time"] += processing_duration.total_seconds()

            stats = {
                "status": "success",
                "original_texts": len(texts),
                "cleaned_texts": len(cleaned_texts),
                "final_processed_texts": len(processed_texts),
                "valid_indices_count": len(valid_indices),
                "summarization_enabled": self.text_config["enable_summarization"],
                "processing_duration_seconds": processing_duration.total_seconds()
            }

            logging.info("Text processing completed: %d processed texts from %d original",
                        len(processed_texts), len(texts))

            return processed_texts, valid_indices, stats

        except Exception as e:
            processing_duration = datetime.now() - processing_start
            logging.error("Text processing failed: %s", str(e))

            stats = {
                "status": "failed",
                "error": str(e),
                "processing_duration_seconds": processing_duration.total_seconds()
            }

            return pd.Series([]), [], stats

    async def process_texts_for_prediction(self, texts: List[str]) -> Tuple[pd.Series, List[int], Dict]:
        """
        Process texts for real-time prediction (optimized for speed).

        Args:
            texts: List of texts to process

        Returns:
            Tuple of (processed_texts, valid_indices, processing_stats)
        """
        processing_start = datetime.now()

        try:
            # Clean texts (faster version)
            cleaned_texts, valid_indices = self._clean_and_filter_texts(texts, fast_mode=True)

            if len(cleaned_texts) == 0:
                return pd.Series([]), [], {"status": "failed", "reason": "no_valid_texts"}

            # For prediction, use lighter summarization or skip it for speed
            if self.text_config["enable_summarization"] and len(cleaned_texts) <= 5:
                # Only summarize small batches for prediction
                summarized_texts = await self._summarize_texts_batch(cleaned_texts, batch_size=5)
                processed_texts = pd.Series(summarized_texts, name='processed_text')
            else:
                # Skip summarization for larger batches to maintain speed
                processed_texts = pd.Series(cleaned_texts, name='processed_text')

            processing_duration = datetime.now() - processing_start

            stats = {
                "status": "success",
                "texts_processed": len(processed_texts),
                "processing_duration_seconds": processing_duration.total_seconds(),
                "summarization_applied": self.text_config["enable_summarization"] and len(cleaned_texts) <= 5
            }

            return processed_texts, valid_indices, stats

        except Exception as e:
            processing_duration = datetime.now() - processing_start
            logging.error("Prediction text processing failed: %s", str(e))

            return pd.Series([]), [], {
                "status": "failed",
                "error": str(e),
                "processing_duration_seconds": processing_duration.total_seconds()
            }

    def _clean_and_filter_texts(self, texts: List[str], fast_mode: bool = False) -> Tuple[List[str], List[int]]:
        """Clean and filter texts"""
        cleaned_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                continue

            # Basic cleaning
            cleaned = text.strip()

            if not fast_mode and self.text_config["clean_text"]:
                cleaned = self._deep_clean_text(cleaned)
            elif fast_mode:
                cleaned = self._fast_clean_text(cleaned)

            # Filter by minimum length
            if len(cleaned) >= self.text_config["min_text_length"]:
                # Truncate if too long
                max_length = self.text_config["max_input_length"]
                if len(cleaned) > max_length:
                    cleaned = cleaned[:max_length] + "..."

                cleaned_texts.append(cleaned)
                valid_indices.append(i)
                self.processing_stats["texts_cleaned"] += 1

        logging.debug("Cleaned %d texts from %d original", len(cleaned_texts), len(texts))
        return cleaned_texts, valid_indices

    def _deep_clean_text(self, text: str) -> str:
        """Comprehensive text cleaning"""
        # Remove URLs
        if self.text_config["remove_urls"]:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        if self.text_config["remove_emails"]:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Normalize whitespace
        if self.text_config["normalize_whitespace"]:
            text = re.sub(r'\s+', ' ', text)

        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)

        # Remove excessive punctuation
        text = re.sub(r'[.!?]{3,}', '...', text)

        return text.strip()

    def _fast_clean_text(self, text: str) -> str:
        """Fast text cleaning for prediction"""
        # Basic whitespace normalization only
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def _summarize_texts_batch(self, texts: List[str], batch_size: int) -> List[str]:
        """Summarize texts in batches using Azure OpenAI"""
        summarized_texts = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        print(f"Processing {len(texts)} texts in {total_batches} batches of {batch_size}")
        start_time = datetime.now()

        # Process batches with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(10)  # Process up to 10 batches concurrently
        tasks = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_start_idx = i

            # Add batch processing task with batch index for ordering
            tasks.append(self._process_batch_with_semaphore(semaphore, batch_texts, batch_start_idx, i // batch_size))

        # Wait for all batches to complete and get results in order
        batch_results = await asyncio.gather(*tasks)

        # Sort results by batch index and flatten the list
        batch_results.sort(key=lambda x: x[0])
        for _, batch_summaries in batch_results:
            summarized_texts.extend(batch_summaries)

        elapsed = (datetime.now() - start_time).total_seconds()
        texts_per_second = len(texts) / elapsed if elapsed > 0 else 0
        print(f"Summarization completed in {elapsed:.1f} seconds ({texts_per_second:.1f} texts/sec)")

        return summarized_texts

    async def _process_batch_with_semaphore(self, semaphore, batch_texts, batch_start_idx, batch_idx):
        """Process a batch with semaphore to limit concurrency"""
        async with semaphore:
            try:
                start_time = datetime.now()
                batch_summaries = await self._summarize_batch(batch_texts, batch_start_idx)
                elapsed = (datetime.now() - start_time).total_seconds()

                # Log progress for each batch
                if batch_idx % 5 == 0:  # Log every 5 batches
                    print(f"  ✓ Batch {batch_idx+1}: Processed {len(batch_summaries)} texts in {elapsed:.1f}s")

                return batch_idx, batch_summaries

            except Exception as e:
                logging.error(f"Batch {batch_idx+1} failed: {str(e)}")
                # Return original texts as fallback
                return batch_idx, batch_texts

    async def _summarize_batch(self, texts: List[str], batch_start_idx: int) -> List[str]:
        """Summarize a single batch of texts"""
        summaries = []

        for i, text in enumerate(texts):
            try:
                summary = await self._summarize_single_text(text)
                summaries.append(summary)
                self.processing_stats["texts_summarized"] += 1

            except Exception as e:
                logging.warning("Summarization failed for text %d: %s",
                              batch_start_idx + i, str(e))
                # Fallback to original text
                summaries.append(text)
                self.processing_stats["summarization_failures"] += 1

        return summaries

    async def _summarize_single_text(self, text: str) -> str:
        """Summarize a single text using Azure OpenAI"""
        # Skip summarization for very short texts (improved efficiency)
        if not text or len(text) < 100:
            return text

        # Check if client is initialized
        if self.client is None:
            return text

        # Use self.max_retries which we now initialize in __init__
        for attempt in range(self.max_retries):
            try:
                # Create prompt from template
                summarization_config = getattr(self.config.preprocessing, 'summarization', {})
                prompt_template = summarization_config.get('summary_prompt_template',
                    "Summarize the following incident information in 30 words or less: {combined_text}")
                model_name = summarization_config.get('model_name', "gpt-35-turbo")

                prompt = prompt_template.replace("{combined_text}", text)

                # Send OpenAI request with timeout to avoid hanging
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You summarize IT incident tickets concisely."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=60,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    ),
                    timeout=25.0  # 25 second timeout for API calls (fixed comment)
                )

                # Extract summary from response
                summary = response.choices[0].message.content.strip()
                return summary

            except asyncio.TimeoutError:
                logging.warning(f"Summarization timed out on attempt {attempt+1}/{self.max_retries}")
                if attempt >= self.max_retries - 1:
                    return text
                await asyncio.sleep(1.0)  # 1 second pause before retry

            except Exception as e:
                if attempt >= self.max_retries - 1:
                    return text
                # Wait before retrying (with exponential backoff)
                retry_wait = self.retry_delay * (2 ** attempt)  # Exponential backoff starting at self.retry_delay (1.0)
                await asyncio.sleep(retry_wait)

        return text

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using Azure OpenAI

        Args:
            texts: List of text strings to generate embeddings for

        Returns:
            List of embeddings (float arrays)
        """
        try:
            print(f"TextProcessor: Starting embedding generation for {len(texts)} texts")

            # Initialize embedding generator
            from preprocessing.embedding_generation import EmbeddingGenerator

            print("TextProcessor: Initialized EmbeddingGenerator class")

            # Initialize embedding generator
            embedding_generator = EmbeddingGenerator(self.config)

            print(f"TextProcessor: Created embedding generator with config: {embedding_generator.client is not None}")

            # Try a simple test with a single text
            test_text = "This is a test text for embedding generation."
            print("TextProcessor: Testing with a single embedding...")
            test_result = await embedding_generator.generate_single_embedding(test_text)
            if test_result:
                print(f"TextProcessor: Test embedding generated successfully. First few values: {test_result[:5]}")
            else:
                print("TextProcessor: Test embedding failed!")

            # Generate embeddings using the specialized class
            print("TextProcessor: Beginning batch embedding generation...")

            embeddings_array, valid_indices = await embedding_generator.generate_embeddings_batch(
                texts, batch_size=50
            )

            # Convert numpy array to list of lists for consistency
            if len(embeddings_array) > 0:
                embeddings_list = [embedding.tolist() for embedding in embeddings_array]
                print(f"TextProcessor: Successfully generated {len(embeddings_list)} embeddings")

                # Show first few values of the first embedding
                if embeddings_list:
                    first_embedding = embeddings_list[0]
                    print(f"TextProcessor: First embedding values: {first_embedding[:5]}")

                return embeddings_list
            else:
                print("TextProcessor: No embeddings were generated. Falling back to zeros.")
                return [[0.0] * 1536] * len(texts)  # Default OpenAI embedding dimension
        except Exception as e:
            print(f"TextProcessor: Failed to generate embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return zero vectors as fallback
            return [[0.0] * 1536] * len(texts)  # Default OpenAI embedding dimension

    def _create_output_dataframe(self, input_df: pd.DataFrame, summaries: pd.Series,
                                embeddings: List[List[float]], valid_indices: List[int]) -> pd.DataFrame:
        """Create output DataFrame with only the required columns"""

        # Get processing version from config
        processing_version = getattr(self.config.preprocessing, 'processing_version', 'v2.0.0')
        current_timestamp = datetime.now()

        # Create output DataFrame with only valid rows
        valid_rows = input_df.iloc[valid_indices].copy()

        output_data = {
            'number': valid_rows['number'].values,
            'sys_created_on': valid_rows['sys_created_on'].values,
            'combined_incidents_summary': summaries.values,
            'embedding': embeddings,
            'created_timestamp': [current_timestamp] * len(valid_rows),
            'processing_version': [processing_version] * len(valid_rows)
        }

        output_df = pd.DataFrame(output_data)

        logging.info("Created output DataFrame with %d rows and columns: %s",
                    len(output_df), list(output_df.columns))

        return output_df

    async def process_incident_for_embedding_batch(self, df: pd.DataFrame,
                                           batch_size: int = 10) -> Tuple[pd.DataFrame, Dict]:
        """
        Process incident DataFrame and return only the required output columns.

        Args:
            df: DataFrame with incident columns (number, sys_created_on, description, short_description, business_service)
            batch_size: Batch size for summarization

        Returns:
            Tuple of (output_dataframe, processing_stats)
        """
        processing_start = datetime.now()

        logging.info("Processing %d incidents for embedding", len(df))

        try:
            # Step 1: Validate required input columns
            required_cols = ['number', 'sys_created_on', 'description', 'short_description', 'business_service']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Step 2: Combine and clean text columns
            combined_texts = self._combine_incident_columns(df)

            # Step 3: Process combined texts for summarization
            processed_texts, valid_indices, processing_stats = await self.process_texts_for_training(
                combined_texts, batch_size=batch_size
            )

            # Step 4: Generate embeddings from summaries
            embeddings_data = await self._generate_embeddings_batch(processed_texts.tolist())

            # Step 5: Create output DataFrame with only required columns
            output_df = self._create_output_dataframe(df, processed_texts, embeddings_data, valid_indices)

            processing_duration = datetime.now() - processing_start

            stats = {
                "status": "success",
                "total_incidents": len(df),
                "processed_incidents": len(output_df),
                "success_rate": (len(output_df) / len(df)) * 100 if len(df) > 0 else 0,
                "processing_duration_seconds": processing_duration.total_seconds(),
                "failed_incidents": len(df) - len(output_df),
                "text_processing_stats": processing_stats
            }

            logging.info("Incident processing completed: %.1f%% success rate (%d/%d)",
                        stats['success_rate'], stats['processed_incidents'], stats['total_incidents'])

            return output_df, stats

        except Exception as e:
            processing_duration = datetime.now() - processing_start
            logging.error("Incident processing failed: %s", str(e))

            # Return empty DataFrame with correct structure
            empty_df = pd.DataFrame(columns=[
                'number', 'sys_created_on', 'combined_incidents_summary',
                'embedding', 'created_timestamp', 'processing_version'
            ])

            stats = {
                "status": "failed",
                "error": str(e),
                "total_incidents": len(df),
                "processed_incidents": 0,
                "success_rate": 0,
                "processing_duration_seconds": processing_duration.total_seconds()
            }

            return empty_df, stats

    def _combine_incident_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Combine and clean the configured text columns from incident DataFrame.

        Args:
            df: DataFrame with incident data

        Returns:
            List of combined and cleaned text strings
        """
        text_columns = self.config.preprocessing.text_columns_to_process
        combined_texts = []

        for _, row in df.iterrows():
            text_parts = []

            # Extract and clean each configured column
            for column in text_columns:
                if column in df.columns and pd.notna(row[column]):
                    # Clean the individual column
                    if column == 'business_service':
                        # Use specialized cleaning for business service
                        cleaned_text = self._clean_business_service(str(row[column]))
                    else:
                        # Regular cleaning for other columns
                        cleaned_text = self._deep_clean_text(str(row[column]))

                    if len(cleaned_text.strip()) > 0:
                        text_parts.append(cleaned_text)

            # Combine all parts with separator
            if text_parts:
                combined_text = " | ".join(text_parts)
                combined_texts.append(combined_text)
            else:
                # Handle case where no valid text found
                combined_texts.append("No description available")

        logging.debug("Combined %d incident texts from columns: %s",
                     len(combined_texts), text_columns)
        return combined_texts

    def _clean_business_service(self, text: str) -> str:
        """
        Clean business service names by removing environment suffixes and standardizing format

        Args:
            text: Raw business service name

        Returns:
            Cleaned business service name
        """
        if not text or not isinstance(text, str):
            return ""

        # Option 1: Keep environment indicators (comment out these lines if you want to keep PROD/TEST/DEV)
        env_patterns = [
            r'\s*-\s*(PROD|TEST|DEV)(\s|$)',  # Remove "- PROD", "- TEST", "- DEV"
            r'\s+PROD$', r'\s+TEST$', r'\s+DEV$'  # Remove terminal "PROD", "TEST", "DEV"
        ]

        cleaned = text.strip()
        for pattern in env_patterns:
            cleaned = re.sub(pattern, ' ', cleaned)

        # Remove special characters completely (instead of replacing with space)
        cleaned = re.sub(r'[-_\+]', '', cleaned)

        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Remove any trailing/leading special chars and whitespace
        cleaned = re.sub(r'^[-\s]+|[-\s]+$', '', cleaned)

        return cleaned.strip()

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get text processing statistics"""
        stats = self.processing_stats.copy()

        # Calculate derived metrics
        if stats["texts_processed"] > 0:
            stats["cleaning_success_rate"] = (stats["texts_cleaned"] / stats["texts_processed"]) * 100

        if stats["texts_summarized"] + stats["summarization_failures"] > 0:
            stats["summarization_success_rate"] = (
                stats["texts_summarized"] /
                (stats["texts_summarized"] + stats["summarization_failures"]) * 100
            )
        else:
            stats["summarization_success_rate"] = 0

        return stats

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate text processing configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }

        # Check Azure OpenAI configuration if summarization is enabled
        if self.text_config["enable_summarization"]:
            azure_config = self.config.azure.openai

            if not azure_config.get('endpoint'):
                validation_results["errors"].append("Azure OpenAI endpoint required for summarization")
                validation_results["valid"] = False

            if not azure_config.get('api_key'):
                validation_results["errors"].append("Azure OpenAI API key required for summarization")
                validation_results["valid"] = False

            if not azure_config.get('summarization_model'):
                validation_results["warnings"].append("Summarization model not specified, using default")

        # Check configuration values
        if self.text_config["min_text_length"] <= 0:
            validation_results["errors"].append("Minimum text length must be positive")
            validation_results["valid"] = False

        if self.text_config["max_input_length"] <= self.text_config["min_text_length"]:
            validation_results["errors"].append("Maximum input length must be greater than minimum")
            validation_results["valid"] = False

        return validation_results

    def reset_statistics(self):
        """Reset processing statistics"""
        self.processing_stats = {
            "texts_processed": 0,
            "texts_cleaned": 0,
            "texts_summarized": 0,
            "summarization_failures": 0,
            "total_processing_time": 0
        }
        logging.info("Text processor statistics reset")