import re
import logging
import pandas as pd
import openai
import time
import uuid
from typing import Tuple
from tqdm import tqdm

class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.openai_client = self._setup_openai_client()
    
    def _setup_openai_client(self):
        """Setup Azure OpenAI client"""
        return openai.AzureOpenAI(
            azure_endpoint=self.config.azure.openai_endpoint,
            api_key=self.config.azure.openai_api_key,
            api_version=self.config.azure.api_version,
        )
    
    def clean_text_for_summary(self, text: str) -> str:
        """Clean text by normalizing whitespace and removing special characters"""
        if not text:
            return ""
        
        # Replace newlines and tabs with spaces
        text = re.sub(r'[\n\r\t]+', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c.isspace())
        
        return text.strip()
    
    def normalize_business_service(self, text: str) -> str:
        """Normalize business service names"""
        if not text:
            return ""
        
        # Remove trailing " - PROD", " - DEV", etc.
        text = re.sub(r'\s*[-_]\s*(PROD|DEV|TEST|UAT|QA)$', '', text)
        
        # Replace dashes and underscores with spaces
        text = re.sub(r'[-_]+', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_safe_text(self, row, column: str) -> str:
        """Helper function to safely get text from a DataFrame row column"""
        return str(row[column]) if pd.notna(row[column]) else ""
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text"""
        if not text:
            return 0
        # OpenAI models generally use ~4 chars per token for English text
        return len(text) // 4 + 1
    
    def summarize_incident_with_llm(self, short_desc: str, desc: str, business_service: str) -> str:
        """Use Azure OpenAI to generate a concise summary of the incident"""
        # Construct the prompt for summarization
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

Example summaries:
- "Adobe Campaign platform prevents users from publishing and unpublishing content due to authorization errors"
- "SAP system fails to process financial transactions with timeout errors during month-end closing"
- "Microsoft Teams meetings crash when screen sharing is attempted from macOS devices"
"""
        
        # Call Azure OpenAI with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.azure.chat_model,
                    messages=[
                        {"role": "system", "content": "You are a technical IT incident summarizer that creates precise, concise summaries focusing on affected systems and specific issues."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=100,
                    timeout=15
                )
                summary = response.choices[0].message.content.strip()
                # Remove any quotes that might be in the response
                summary = summary.strip('"\'')
                return summary
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logging.warning(f"Summarization attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All summarization attempts failed: {e}")
                    # Fall back to simply returning the short description
                    return short_desc or "No description available"
    
    def process_incident_for_embedding_batch(self, df: pd.DataFrame, batch_size: int = 10) -> Tuple[pd.Series, dict]:
        """Process incident text for embedding with LLM summarization in batches"""
        result_series = pd.Series(index=df.index)
        num_batches = (len(df) + batch_size - 1) // batch_size
        
        # Initialize fallback tracking statistics
        fallback_stats = {
            "total_incidents": len(df),
            "short_desc_fallbacks": 0,
            "api_failure_fallbacks": 0,
            "final_sweep_fallbacks": 0,
            "llm_processed": 0
        }
        
        # Initialize progress bar
        pbar = tqdm(total=len(df), desc="Summarizing incidents")
        
        # Process in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # Prepare batch data for API call
            batch_items = []
            for idx, row in batch_df.iterrows():
                # Get text fields
                short_desc = self.get_safe_text(row, 'short_description')
                desc = self.get_safe_text(row, 'description')
                business_svc = self.get_safe_text(row, 'business_service')
                
                # Clean and normalize
                clean_short_desc = self.clean_text_for_summary(short_desc)
                clean_desc = self.clean_text_for_summary(desc)
                clean_desc = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', clean_desc)
                clean_business_svc = self.normalize_business_service(business_svc)
                
                # Use simpler approach for short descriptions
                if len(clean_desc) < 300:
                    result_series[idx] = f"{clean_short_desc} - {clean_business_svc}".strip()
                    fallback_stats["short_desc_fallbacks"] += 1
                    pbar.update(1)
                    continue
                    
                # For longer texts, add to batch for summarization
                batch_items.append({
                    "idx": idx,
                    "short_desc": clean_short_desc,
                    "description": clean_desc[:1500],  # Truncate long descriptions
                    "business_service": clean_business_svc
                })
            
            # If batch is empty after filtering short descriptions, continue
            if not batch_items:
                continue
            
            # Count number of items going to LLM for summarization
            fallback_stats["llm_processed"] += len(batch_items)
                
            # Create prompt for batch summarization
            batch_request_id = str(uuid.uuid4())
            prompt = "Summarize each of these IT incidents in a single concise sentence (max 30 words).\n\n"
            
            for i, item in enumerate(batch_items):
                prompt += f"INCIDENT #{i+1}:\n"
                prompt += f"SHORT DESCRIPTION: {item['short_desc']}\n"
                prompt += f"DESCRIPTION: {item['description']}\n"
                prompt += f"AFFECTED SYSTEM: {item['business_service']}\n\n"
                
            prompt += """For each incident, your summary MUST:
1. Begin by clearly identifying the affected platform/application
2. Describe the specific technical issue
3. Be technical but clear and concise (one sentence only)

RESPOND IN JSON FORMAT:
{
  "summaries": [
    {"incident": 1, "summary": "Platform X fails with error Y when performing action Z"},
    {"incident": 2, "summary": "..."}
    // etc. for each incident
  ]
}
"""
            
            # Make batch API call with retry logic
            max_retries = 3
            batch_failures = 0
            for attempt in range(max_retries):
                try:
                    correlation_id = str(uuid.uuid4())
                        
                    response = self.openai_client.chat.completions.create(
                        model=self.config.azure.chat_model,
                        messages=[
                            {"role": "system", "content": "You are a technical IT incident summarizer that creates precise, concise summaries. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"},
                        timeout=60,
                        user=correlation_id
                    )
                    
                    # Parse response
                    import json
                    response_text = response.choices[0].message.content
                    result = json.loads(response_text)
                    
                    # Validate structure
                    if "summaries" not in result or not isinstance(result["summaries"], list):
                        raise ValueError("Response missing required 'summaries' array")
                        
                    # Map summaries back to original indices
                    for i, summary_obj in enumerate(result.get("summaries", [])):
                        if i < len(batch_items) and "summary" in summary_obj:
                            idx = batch_items[i]["idx"]
                            summary = summary_obj.get("summary", "")
                            business_svc = batch_items[i]["business_service"]
                            
                            # If business service isn't in summary, append it
                            if business_svc and business_svc.lower() not in summary.lower():
                                summary = f"{summary} - {business_svc}"
                                
                            result_series[idx] = summary
                    
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = min(30, 2 ** attempt * 2)
                        logging.warning(f"Batch summarization attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"All batch summarization attempts failed: {e}")
                        
                        # Fall back to simple combination for failed items
                        for item in batch_items:
                            idx = item["idx"]
                            if idx not in result_series or pd.isna(result_series[idx]):
                                result_series[idx] = f"{item['short_desc']} - {item['business_service']}".strip()
                                batch_failures += 1
                        
                        fallback_stats["api_failure_fallbacks"] += batch_failures
            
            # Update progress
            pbar.update(min(batch_size, len(batch_df)))
        
        pbar.close()
        
        # Final sweep - fill any missing values with their short description
        final_sweep_count = 0
        for idx in df.index:
            if idx not in result_series or pd.isna(result_series[idx]):
                short_desc = self.get_safe_text(df.loc[idx], 'short_description')
                business_svc = self.get_safe_text(df.loc[idx], 'business_service')
                business_svc = self.normalize_business_service(business_svc)
                result_series[idx] = f"{short_desc} - {business_svc}".strip()
                final_sweep_count += 1
        
        fallback_stats["final_sweep_fallbacks"] = final_sweep_count
        
        # Calculate success rate
        fallback_stats["total_fallbacks"] = (
            fallback_stats["short_desc_fallbacks"] +
            fallback_stats["api_failure_fallbacks"] +
            fallback_stats["final_sweep_fallbacks"]
        )
        fallback_stats["llm_success_count"] = fallback_stats["llm_processed"] - fallback_stats["api_failure_fallbacks"] - fallback_stats["final_sweep_fallbacks"]
        fallback_stats["llm_success_rate"] = fallback_stats["llm_success_count"] / max(1, fallback_stats["llm_processed"]) * 100
        fallback_stats["overall_llm_rate"] = fallback_stats["llm_success_count"] / len(df) * 100
        
        logging.info(f"Summarization fallback stats: {fallback_stats}")
        
        return result_series, fallback_stats