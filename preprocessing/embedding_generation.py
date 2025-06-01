import logging
import time
import json
import requests
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import openai
from openai import AzureOpenAI
import os

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        
        # Get Azure OpenAI configuration
        azure_config = config.get('azure', {})
        openai_config = azure_config.get('openai', {})
        
        # Initialize Azure OpenAI client for embeddings
        embedding_endpoint = openai_config.get('embedding_endpoint', '')
        if '/openai/deployments' in embedding_endpoint:
            base_endpoint = embedding_endpoint.split('/openai/deployments')[0]
        else:
            base_endpoint = embedding_endpoint
            
        self.embedding_client = AzureOpenAI(
            api_key=openai_config.get('embedding_key', ''),
            api_version=openai_config.get('embedding_api_version', ''),
            azure_endpoint=base_endpoint
        )
        
        # Initialize Azure OpenAI client for chat (summaries)
        self.chat_client = AzureOpenAI(
            api_key=openai_config.get('api_key', ''),
            api_version=openai_config.get('api_version', ''),
            azure_endpoint=openai_config.get('endpoint', '')
        )
        
        self.embedding_model = openai_config.get('embedding_model', 'text-embedding-3-large')
        self.chat_deployment = openai_config.get('deployment_name', 'gpt-4o')
        
        # Embedding configuration - Pure semantic as requested
        clustering_config = config.get('clustering', {})
        self.embedding_weights = clustering_config.get('embedding', {}).get('weights', {
            'semantic': 1.0, 'entity': 0.0, 'action': 0.0
        })
        self.batch_size = clustering_config.get('embedding', {}).get('batch_size', 25)
        
        logging.info(f"EmbeddingGenerator initialized with model: {self.embedding_model}")
        logging.info(f"Embedding weights: {self.embedding_weights}")
    
    def generate_incident_summary(self, short_description: str, description: str, max_retries: int = 3) -> str:
        """Generate AI-powered incident summary using Azure OpenAI"""
        
        # Combine short description and full description
        combined_text = f"Title: {short_description}\nDetails: {description}"
        
        prompt = f"""
        Analyze this IT incident and create a concise, technical summary focusing on the core issue:

        {combined_text}

        Instructions:
        - Identify the main technical problem or issue
        - Extract key technical components (systems, applications, errors)
        - Remove unnecessary details and user-specific information
        - Focus on the root cause or symptom
        - Keep it under 100 words
        - Use technical terminology where appropriate

        Summary:
        """
        
        for attempt in range(max_retries):
            try:
                response = self.chat_client.chat.completions.create(
                    model=self.chat_deployment,
                    messages=[
                        {"role": "system", "content": "You are an IT incident analyst who creates concise technical summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=150
                )
                
                summary = response.choices[0].message.content.strip()
                
                # Clean up the summary
                if summary.startswith("Summary:"):
                    summary = summary[8:].strip()
                
                return summary
                
            except Exception as e:
                logging.warning(f"Summary generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"Failed to generate summary after {max_retries} attempts")
                    # Fallback to simple text processing
                    return self._simple_summary_fallback(short_description, description)
    
    def _simple_summary_fallback(self, short_description: str, description: str) -> str:
        """Simple fallback summary when AI generation fails"""
        # Take first 150 characters of combined text
        combined = f"{short_description}. {description}"
        if len(combined) <= 150:
            return combined
        return combined[:147] + "..."
    
    def generate_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Generate embedding for a single text using Azure OpenAI"""
        
        for attempt in range(max_retries):
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                
                embedding = response.data[0].embedding
                
                # Apply embedding weights (pure semantic as configured)
                if self.embedding_weights['semantic'] != 1.0:
                    # Scale embedding if semantic weight is not 1.0
                    embedding = [x * self.embedding_weights['semantic'] for x in embedding]
                
                return embedding
                
            except Exception as e:
                logging.warning(f"Embedding generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"Failed to generate embedding after {max_retries} attempts")
                    return None
    
    def generate_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        total_texts = len(texts)
        
        # Process in batches to avoid rate limits
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for j, text in enumerate(batch):
                if show_progress and (i + j) % 10 == 0:
                    logging.info(f"Processing embedding {i + j + 1}/{total_texts}")
                
                embedding = self.generate_embedding(text)
                batch_embeddings.append(embedding)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            embeddings.extend(batch_embeddings)
            
            # Longer delay between batches
            if i + self.batch_size < total_texts:
                time.sleep(1)
        
        return embeddings
    
    def process_incidents_with_embeddings(self, incidents_df: pd.DataFrame) -> pd.DataFrame:
        """Process incidents to generate summaries and embeddings"""
        
        processed_incidents = incidents_df.copy()
        
        # Generate AI summaries
        logging.info("Generating AI-powered incident summaries...")
        summaries = []
        
        for idx, row in incidents_df.iterrows():
            if idx % 10 == 0:
                logging.info(f"Processing summary {idx + 1}/{len(incidents_df)}")
            
            summary = self.generate_incident_summary(
                row.get('short_description', ''),
                row.get('description', '')
            )
            summaries.append(summary)
            
            # Small delay to avoid rate limiting
            time.sleep(0.2)
        
        processed_incidents['combined_incidents_summary'] = summaries
        
        # Generate embeddings from summaries
        logging.info("Generating semantic embeddings...")
        embeddings = self.generate_embeddings_batch(summaries)
        
        # Filter out failed embeddings
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        
        if len(valid_indices) < len(embeddings):
            logging.warning(f"Failed to generate {len(embeddings) - len(valid_indices)} embeddings")
            processed_incidents = processed_incidents.iloc[valid_indices].copy()
            embeddings = [embeddings[i] for i in valid_indices]
        
        processed_incidents['embedding'] = embeddings
        processed_incidents['processed_timestamp'] = pd.Timestamp.now()
        
        logging.info(f"Successfully processed {len(processed_incidents)} incidents with embeddings")
        
        return processed_incidents
    
    def validate_embeddings(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Validate embedding quality and return metrics"""
        if not embeddings:
            return {"valid": False, "error": "No embeddings provided"}
        
        # Convert to numpy array for analysis
        embedding_matrix = np.array(embeddings)
        
        # Check for NaN or infinite values
        has_nan = np.any(np.isnan(embedding_matrix))
        has_inf = np.any(np.isinf(embedding_matrix))
        
        if has_nan or has_inf:
            return {
                "valid": False, 
                "error": f"Invalid values detected - NaN: {has_nan}, Inf: {has_inf}"
            }
        
        # Calculate basic statistics
        dimensions = embedding_matrix.shape[1]
        variance = np.var(embedding_matrix)
        mean_norm = np.mean(np.linalg.norm(embedding_matrix, axis=1))
        
        return {
            "valid": True,
            "count": len(embeddings),
            "dimensions": dimensions,
            "variance": float(variance),
            "mean_norm": float(mean_norm),
            "embedding_weights_used": self.embedding_weights
        }