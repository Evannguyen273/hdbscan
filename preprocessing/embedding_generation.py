import logging
import numpy as np
import pandas as pd
import openai
import time
import uuid
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Tuple, Dict, Any

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.embedding_client = self._setup_embedding_client()
    
    def _setup_embedding_client(self):
        """Setup Azure OpenAI embedding client"""
        return openai.AzureOpenAI(
            api_key=self.config.azure.openai_embedding_key,
            api_version=self.config.azure.api_version,
            azure_endpoint=self.config.azure.openai_embedding_endpoint
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text"""
        if not text:
            return 0
        return len(text) // 4 + 1
    
    def classify_terms_with_llm(self, df: pd.DataFrame, text_column: str = 'combined_incidents_summary') -> Dict[str, Dict]:
        """Use GPT to classify terms from incidents into entities and actions"""
        # Combine all text for term extraction
        sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
        all_text = " ".join(sample_df[text_column].fillna('').astype(str).tolist())
        
        prompt = f"""
Analyze the following IT incidents text and identify:
1. ENTITIES: Technical components, systems, software, or services mentioned (like SAP, Outlook, VPN)
2. ACTIONS: Verbs describing what happened or needs to happen (like crashed, failed, reset)

Extract up to 50 most common entities and up to 50 most common actions.

YOU MUST RESPOND WITH VALID JSON using this structure:
{{
    "ENTITY": {{"term1": frequency, "term2": frequency, ...}},
    "ACTION": {{"term1": frequency, "term2": frequency, ...}}
}}

Incident text sample:
{all_text[:3000]}
"""
        
        correlation_id = str(uuid.uuid4())
        retry_attempts = 3
        
        for attempt in range(retry_attempts):
            try:
                response = self.embedding_client.chat.completions.create(
                    model=self.config.azure.chat_model,
                    messages=[
                        {"role": "system", "content": "You are an expert IT analyst that extracts and classifies terms from incident descriptions. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    timeout=30,
                    user=correlation_id
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Validate expected structure
                if "ENTITY" in result and "ACTION" in result:
                    return result
                else:
                    raise ValueError("Response missing required ENTITY or ACTION keys")
                    
            except Exception as e:
                if attempt < retry_attempts - 1:
                    wait_time = min(30, 2 ** attempt * 2)
                    logging.warning(f"Term classification attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All term classification attempts failed: {e}")
                    return {"ENTITY": {}, "ACTION": {}}
    
    def generate_semantic_embeddings(self, text_series: pd.Series, batch_size: int = 25) -> np.ndarray:
        """Generate semantic embeddings using Azure OpenAI"""
        semantic_embeddings = []
        total_batches = (len(text_series) + batch_size - 1) // batch_size
        retry_limit = 3
        
        # Azure-specific metrics tracking
        azure_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "total_time": 0
        }
        
        # Reduce batch size for very large texts
        max_estimated_tokens = text_series.apply(lambda x: self.estimate_tokens(str(x))).max()
        if max_estimated_tokens > 4000:
            adjusted_batch_size = max(1, min(batch_size, 5))
            logging.warning(f"Detected large texts (est. {max_estimated_tokens} tokens). Reducing batch size to {adjusted_batch_size}")
            batch_size = adjusted_batch_size
        
        # Process in batches
        with tqdm(total=len(text_series), desc="Embedding texts") as pbar:
            for i in range(0, len(text_series), batch_size):
                batch = text_series.iloc[i:i+batch_size].fillna('').tolist()
                batch_size_actual = len(batch)
                correlation_id = str(uuid.uuid4())
                
                azure_metrics["total_requests"] += 1
                start_time = time.time()
                
                try:
                    response = self.embedding_client.embeddings.create(
                        input=batch,
                        model=self.config.azure.embedding_model,
                        timeout=30,
                        user=correlation_id
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]
                    semantic_embeddings.extend(batch_embeddings)
                    
                    # Update metrics
                    azure_metrics["successful_requests"] += 1
                    azure_metrics["total_time"] += time.time() - start_time
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Handle rate limiting
                    if "rate limit" in error_msg or "too many requests" in error_msg:
                        azure_metrics["rate_limited_requests"] += 1
                        wait_time = min(60, 2 ** (azure_metrics["rate_limited_requests"] % 6))
                        logging.warning(f"Rate limit hit. Backing off for {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    
                    # Handle token limit errors
                    elif "token" in error_msg or "context length" in error_msg:
                        if batch_size > 1:
                            new_batch_size = max(1, batch_size // 2)
                            logging.warning(f"Token limit exceeded. Reducing batch size to {new_batch_size}")
                            batch_size = new_batch_size
                            continue
                        else:
                            logging.error(f"Single item too long: {len(text_series.iloc[i])}")
                            semantic_embeddings.extend([[0.0] * 3072])
                    
                    else:
                        azure_metrics["failed_requests"] += 1
                        logging.error(f"Embedding error: {error_msg}")
                        semantic_embeddings.extend([[0.0] * 3072 for _ in range(batch_size_actual)])
                
                pbar.update(batch_size_actual)
        
        # Log metrics
        avg_time = azure_metrics["total_time"] / max(1, azure_metrics["successful_requests"])
        logging.info(f"Azure OpenAI Embedding API metrics: {azure_metrics['total_requests']} requests, "
                    f"{azure_metrics['successful_requests']} successful, {azure_metrics['failed_requests']} failed, "
                    f"{azure_metrics['rate_limited_requests']} rate limited, avg time: {avg_time:.2f}s")
        
        # Ensure we return the right number of embeddings
        if len(semantic_embeddings) < len(text_series):
            logging.warning(f"Missing embeddings: expected {len(text_series)}, got {len(semantic_embeddings)}. Padding with zeros.")
            semantic_embeddings.extend([[0.0] * 3072 for _ in range(len(text_series) - len(semantic_embeddings))])
        
        return np.array(semantic_embeddings)
    
    def create_hybrid_embeddings(self, df: pd.DataFrame, text_column: str = 'combined_incidents_summary') -> Tuple[pd.DataFrame, Dict, Dict]:
        """Create hybrid embeddings combining entity, action, and semantic information"""
        logging.info(f"Creating hybrid embeddings for {len(df)} records")
        result_df = df.copy()
        
        # Get entity and action classification
        logging.info("Classifying terms into entities and actions...")
        classification_result = self.classify_terms_with_llm(result_df, text_column)
        
        # Extract entity and action terms
        entity_terms = list(classification_result.get("ENTITY", {}).keys())
        action_terms = list(classification_result.get("ACTION", {}).keys())
        logging.info(f"Using {len(entity_terms)} entities and {len(action_terms)} action terms")
        
        # Create vectorizers and generate entity/action embeddings
        vectorizers = {
            'entity': TfidfVectorizer(vocabulary=entity_terms, lowercase=True),
            'action': TfidfVectorizer(vocabulary=action_terms, lowercase=True)
        }
        
        # Generate and transform embeddings
        text_data = result_df[text_column].fillna('')
        matrices = {
            'entity': vectorizers['entity'].fit_transform(text_data),
            'action': vectorizers['action'].fit_transform(text_data)
        }
        
        # Convert sparse matrices to dense
        dense_matrices = {k: m.toarray() for k, m in matrices.items()}
        
        # Generate semantic embeddings
        logging.info("Generating semantic embeddings...")
        semantic_embeddings = self.generate_semantic_embeddings(
            result_df[text_column],
            batch_size=self.config.embedding.batch_size
        )
        
        # Scale all embedding components
        logging.info("Scaling and combining embeddings...")
        scaled_matrices = {}
        for name, matrix in {**dense_matrices, 'semantic': semantic_embeddings}.items():
            scaler = StandardScaler()
            scaled_matrices[name] = scaler.fit_transform(matrix)
        
        # Set weights from config
        weights = {
            'entity': self.config.embedding.entity_weight,
            'action': self.config.embedding.action_weight,
            'semantic': self.config.embedding.semantic_weight
        }
        
        # Get dimensions for each component
        dims = {k: m.shape[1] for k, m in scaled_matrices.items()}
        
        # Create combined embeddings array
        total_dims = sum(dims.values())
        combined_embeddings = np.zeros((len(result_df), total_dims))
        
        # Fill combined embeddings with weighted components
        start_idx = 0
        for name, matrix in scaled_matrices.items():
            end_idx = start_idx + dims[name]
            combined_embeddings[:, start_idx:end_idx] = weights[name] * matrix
            start_idx = end_idx
        
        # Convert embeddings to JSON strings
        result_df['embedding'] = [json.dumps(emb.tolist()) for emb in combined_embeddings]
        
        logging.info("Hybrid embedding generation complete")
        
        # Create fallback stats structure for compatibility
        fallback_stats = {
            "embedding_generation": True,
            "total_embeddings": len(result_df),
            "entity_terms": len(entity_terms),
            "action_terms": len(action_terms)
        }
        
        return result_df, classification_result, fallback_stats