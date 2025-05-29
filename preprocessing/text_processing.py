import logging
import time
import pandas as pd
import re
import json
import uuid
from typing import Tuple, List, Dict, Set
from tqdm.auto import tqdm

class TextProcessor:
    """Enhanced text processing with improved error handling and no fallbacks"""
    
    def __init__(self, config):
        self.config = config
        self.openai_client = self._initialize_openai_client()
        
        # Track failed incident numbers for reporting
        self.failed_incidents: Set[str] = set()
        self.failure_reasons: Dict[str, str] = {}
        
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
    
    def chunk_text_for_summarization(self, text: str, max_input_tokens: int = 100000) -> List[str]:
        """
        Split text into chunks if it's too long for summarization.
        
        Args:
            text: Input text to potentially chunk
            max_input_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= max_input_tokens:
            return [text]
        
        # Split into sentences first
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_input_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence + ". "
                current_tokens += sentence_tokens
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize_text_chunks(self, chunks: List[str], short_desc: str, business_service: str) -> str:
        """
        Summarize multiple text chunks and combine them.
        
        Args:
            chunks: List of text chunks to summarize
            short_desc: Short description
            business_service: Business service
            
        Returns:
            Combined summary
        """
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            prompt = f"""Create a concise summary (max 50 words) of this IT incident part {i+1}/{len(chunks)}:

SHORT DESCRIPTION: {short_desc}
BUSINESS SERVICE: {business_service}
TEXT CHUNK: {chunk}

Focus on: 1) the specific technical issue and 2) key error messages or symptoms.
"""
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a technical IT summarizer. Create precise, concise summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=100,
                    timeout=30
                )
                
                chunk_summary = response.choices[0].message.content.strip().strip('"\'')
                chunk_summaries.append(chunk_summary)
                
            except Exception as e:
                logging.error(f"Failed to summarize chunk {i+1}: {e}")
                raise  # Re-raise to fail the entire summarization
        
        # Combine chunk summaries
        if len(chunk_summaries) == 1:
            return f"{chunk_summaries[0]} - {business_service}"
        else:
            combined = " ".join(chunk_summaries)
            return f"{combined} - {business_service}"
    
    def summarize_incident_with_llm(self, short_desc: str, desc: str, business_service: str, incident_number: str = None) -> str:
        """
        Summarize incident with chunking strategy for long texts.
        
        Args:
            short_desc: Short description
            desc: Full description  
            business_service: Business service
            incident_number: Incident number for error tracking
            
        Returns:
            Summary text
            
        Raises:
            Exception: If summarization fails (no fallbacks)
        """
        # Estimate total tokens needed
        combined_text = f"{short_desc} {desc} {business_service}"
        input_tokens = self.estimate_tokens(combined_text)
        max_output_tokens = 100
        
        # Check if we need chunking (leaving buffer for prompt and response)
        if input_tokens + max_output_tokens > 120000:  # Conservative limit for 128k context
            logging.info(f"Text too long ({input_tokens} tokens), using chunking strategy for incident {incident_number}")
            
            # Chunk the description text
            chunks = self.chunk_text_for_summarization(desc, max_input_tokens=100000)
            return self.summarize_text_chunks(chunks, short_desc, business_service)
        
        # Standard single-call summarization
        prompt = f"""Create a single concise sentence (max 30 words) summarizing this IT incident.
Focus on: 1) the specific platform/application affected and 2) the exact issue or error.

SHORT DESCRIPTION: {short_desc}
DESCRIPTION: {desc}
AFFECTED SYSTEM: {business_service}

Your summary MUST:
1. Begin by clearly identifying the affected platform/application
2. Describe the specific technical issue (not working, error, failure, etc.)
3. Capture any key error messages or symptoms mentioned
4. Be technical but clear and concise (one sentence only)
"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical IT incident summarizer that creates precise, concise summaries focusing on affected systems and specific issues."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100,
            timeout=30
        )
        
        summary = response.choices[0].message.content.strip().strip('"\'')
        
        # Ensure business service is included
        if business_service and business_service.lower() not in summary.lower():
            summary = f"{summary} - {business_service}"
        
        return summary
      def process_incident_for_embedding_batch(self, df: pd.DataFrame, batch_size: int = 10) -> Tuple[pd.Series, Dict]:
        """
        Process incidents for embedding with strict no-fallback policy.
        
        Args:
            df: DataFrame with incidents
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (result_series, statistics)
        """
        logging.info(f"Starting summarization pipeline: {len(df)} incidents, batch_size={batch_size}")
        
        # Reset failure tracking
        self.failed_incidents.clear()
        self.failure_reasons.clear()
        
        result_series = pd.Series(index=df.index, dtype=object)
        successful_count = 0
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        # Process in batches
        with tqdm(total=len(df), desc="Summarizing incidents") as pbar:
            for batch_num in range(0, len(df), batch_size):
                batch_df = df.iloc[batch_num:batch_num+batch_size]
                current_batch = (batch_num // batch_size) + 1
                
                batch_start_time = time.time()
                batch_successes = 0
                batch_failures = 0
                batch_failed_incidents = []
                
                logging.info(f"Processing batch {current_batch}/{total_batches} - incidents {batch_num+1} to {min(batch_num+batch_size, len(df))}")
                
                for idx, row in batch_df.iterrows():
                    incident_number = row.get('number', f'unknown_{idx}')
                    
                    try:
                        # Get and clean text fields
                        short_desc = self._get_safe_text(row, 'short_description')
                        desc = self._get_safe_text(row, 'description')
                        business_svc = self._get_safe_text(row, 'business_service')
                        
                        # Log text length for debugging
                        text_length = len(f"{short_desc} {desc}")
                        estimated_tokens = self.estimate_tokens(f"{short_desc} {desc} {business_svc}")
                        
                        logging.debug(f"Processing {incident_number}: {text_length} chars, ~{estimated_tokens} tokens")
                        
                        # Clean texts
                        short_desc = self._clean_text_for_summary(short_desc)
                        desc = self._clean_text_for_summary(desc)
                        business_svc = self._normalize_business_service(business_svc)
                        
                        # Skip if essential fields are missing
                        if not short_desc and not desc:
                            raise ValueError("No description content available")
                        
                        # Call summarization (will raise exception if it fails)
                        summary = self.summarize_incident_with_llm(
                            short_desc, desc, business_svc, incident_number
                        )
                        
                        result_series[idx] = summary
                        successful_count += 1
                        batch_successes += 1
                        
                        logging.debug(f"✓ {incident_number}: Successfully summarized")
                        
                    except Exception as e:
                        # Track failure but don't create fallback
                        error_msg = str(e)
                        self.failed_incidents.add(incident_number)
                        self.failure_reasons[incident_number] = error_msg
                        batch_failures += 1
                        batch_failed_incidents.append(incident_number)
                        
                        # Classify error type for better logging
                        error_type = self._classify_error(error_msg)
                        logging.warning(f"✗ {incident_number}: {error_type} - {error_msg}")
                        
                    pbar.update(1)
                
                # Log batch completion summary
                batch_duration = time.time() - batch_start_time
                batch_success_rate = (batch_successes / len(batch_df)) * 100 if len(batch_df) > 0 else 0
                
                logging.info(f"Batch {current_batch} complete: {batch_successes}/{len(batch_df)} successful "
                           f"({batch_success_rate:.1f}%) in {batch_duration:.1f}s")
                
                if batch_failed_incidents:
                    logging.warning(f"Batch {current_batch} failures: {batch_failed_incidents}")
                
                # Add small delay between batches to avoid rate limits
                if current_batch < total_batches:
                    time.sleep(0.5)
          # Filter out failed incidents (NaN values)
        result_series = result_series.dropna()
        
        # Create detailed statistics with failure breakdown
        failure_breakdown = {}
        for reason in ["RATE_LIMIT", "TIMEOUT", "TOKEN_LIMIT", "AUTH_ERROR", "CONNECTION_ERROR", "SERVICE_UNAVAILABLE", "API_ERROR"]:
            count = len([k for k, v in self.failure_reasons.items() if reason in self._classify_error(v)])
            if count > 0:
                failure_breakdown[reason] = count
        
        # Create statistics
        stats = {
            "total_incidents": len(df),
            "successful_summarizations": successful_count,
            "failed_incidents": len(self.failed_incidents),
            "success_rate": (successful_count / len(df)) * 100 if len(df) > 0 else 0,
            "failed_incident_numbers": list(self.failed_incidents),
            "failure_breakdown": failure_breakdown,
            "failure_reasons": dict(self.failure_reasons)
        }
        
        logging.info("Summarization complete: %d/%d successful (%.1f%%)", 
                    successful_count, len(df), stats['success_rate'])
        
        if self.failed_incidents:
            logging.warning("Failed incidents: %s", sorted(list(self.failed_incidents)))
            logging.info("Failure breakdown:")
            for error_type, count in failure_breakdown.items():
                logging.info("  - %s: %d incidents", error_type, count)
        
        return result_series, stats
    
    def _get_safe_text(self, row, column: str) -> str:
        """Safely extract text from DataFrame row"""
        value = row.get(column, '')
        return str(value) if pd.notna(value) else ""
    
    def _clean_text_for_summary(self, text: str) -> str:
        """Clean text for summarization"""
        if not text:
            return ""
        
        # Replace newlines and tabs with spaces
        text = re.sub(r'[\n\r\t]+', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove email addresses for privacy
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c.isspace())
        
        return text.strip()
    
    def _normalize_business_service(self, text: str) -> str:
        """Normalize business service names"""
        if not text:
            return ""
        
        # Remove environment suffixes
        text = re.sub(r'\s*[-_]\s*(PROD|DEV|TEST|UAT|QA)$', '', text)
        # Replace dashes and underscores with spaces
        text = re.sub(r'[-_]+', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _classify_error(self, error_msg: str) -> str:
        """Classify error types for better logging"""
        error_msg_lower = error_msg.lower()
        
        if "rate limit" in error_msg_lower or "429" in error_msg_lower:
            return "RATE_LIMIT"
        elif "timeout" in error_msg_lower:
            return "TIMEOUT"
        elif "context length" in error_msg_lower or "token" in error_msg_lower:
            return "TOKEN_LIMIT"
        elif "authentication" in error_msg_lower or "401" in error_msg_lower:
            return "AUTH_ERROR"
        elif "connection" in error_msg_lower:
            return "CONNECTION_ERROR"
        elif "service unavailable" in error_msg_lower or "503" in error_msg_lower:
            return "SERVICE_UNAVAILABLE"
        else:
            return "API_ERROR"
    
    def get_failed_incidents_report(self) -> Dict:
        """Get a report of failed incidents for troubleshooting"""
        return {
            "failed_incident_count": len(self.failed_incidents),
            "failed_incident_numbers": sorted(list(self.failed_incidents)),
            "failure_reasons_summary": {
                reason: len([k for k, v in self.failure_reasons.items() if reason in v])
                for reason in ["rate limit", "token", "timeout", "context length", "api"]
            },
            "detailed_failures": dict(self.failure_reasons)
        }