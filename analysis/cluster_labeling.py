import logging
import json
import time
import uuid
import openai
import os
from typing import Dict, Optional

class ClusterLabeler:
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
    
    def label_clusters_with_llm(
        self, 
        clusters_info: Dict, 
        max_samples: int = 300, 
        chunk_size: int = 25, 
        output_dir: Optional[str] = None
    ) -> Dict:
        """Use Azure OpenAI to generate meaningful labels and descriptions for clusters"""
        
        logging.info(f"Generating cluster labels with LLM for {len(clusters_info)} clusters")
        
        # Skip noise cluster initially
        noise_cluster = clusters_info.get("-1", None)
        clusters_to_label = {k: v for k, v in clusters_info.items() if k != "-1"}
        
        if not clusters_to_label:
            logging.warning("No clusters to label (excluding noise)")
            return {"-1": {"topic": "Noise", "description": "Uncategorized data points"}}
        
        # Process clusters in chunks
        cluster_ids = list(clusters_to_label.keys())
        labeled_clusters = {}
        
        for i in range(0, len(cluster_ids), chunk_size):
            batch_ids = cluster_ids[i:i+chunk_size]
            logging.info(f"Labeling clusters {i} to {i+len(batch_ids)-1}")
            
            # Create prompt for this batch
            prompt = "Analyze these clusters of IT incidents and provide a concise topic name and description for each.\n\n"
            
            for cluster_id in batch_ids:
                cluster = clusters_to_label[cluster_id]
                samples = cluster["samples"][:min(len(cluster["samples"]), max_samples)]
                size = cluster["size"]
                percentage = cluster["percentage"]
                
                prompt += f"CLUSTER {cluster_id} ({size} incidents, {percentage}% of total):\n"
                prompt += "SAMPLES:\n"
                for j, sample in enumerate(samples):
                    prompt += f"{j+1}. {sample}\n"
                prompt += "\n"
            
            prompt += """For each cluster above, provide:
1. A concise topic name (3-7 words) that captures the specific system and issue
2. A brief description (1-2 sentences) that explains the common theme

Format your response as a valid JSON object with this structure:
{
  "cluster_id_1": {
    "topic": "Concise topic name",
    "description": "Brief description of the cluster theme"
  },
  "cluster_id_2": {
    "topic": "Another topic name",
    "description": "Description of another cluster theme"
  }
}

Make your topics specific and technical. Include the affected system name (like SAP, Outlook, VPN) and the specific issue or error where identifiable.
"""
            
            # Call Azure OpenAI with retry logic
            correlation_id = str(uuid.uuid4())
            retry_attempts = 3
            
            for attempt in range(retry_attempts):
                try:
                    logging.info(f"Sending batch of {len(batch_ids)} clusters to LLM (attempt {attempt+1})")
                    
                    response = self.openai_client.chat.completions.create(
                        model=self.config.azure.chat_model,
                        messages=[
                            {"role": "system", "content": "You are an expert IT analyst that provides concise, specific labels for clusters of IT incidents. Always return valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        response_format={"type": "json_object"},
                        timeout=60,
                        user=correlation_id
                    )
                    
                    # Parse response
                    result_text = response.choices[0].message.content
                    batch_labels = json.loads(result_text)
                    
                    # Add to main results
                    if batch_labels:
                        labeled_clusters.update(batch_labels)
                        break
                    else:
                        raise ValueError("Empty response from LLM")
                        
                except Exception as e:
                    if attempt < retry_attempts - 1:
                        wait_time = min(60, 2 ** (attempt + 1))
                        logging.warning(f"Cluster labeling attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"All cluster labeling attempts failed for batch {i}: {e}")
                        
                        # Create fallback labels for this batch
                        for cluster_id in batch_ids:
                            if cluster_id not in labeled_clusters:
                                labeled_clusters[cluster_id] = {
                                    "topic": f"Cluster {cluster_id}",
                                    "description": "Group of related IT incidents"
                                }
        
        # Add noise cluster label
        labeled_clusters["-1"] = {"topic": "Noise", "description": "Uncategorized data points"}
        
        # Save to file if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/labeled_clusters.json", "w") as f:
                json.dump(labeled_clusters, f, indent=2)
            logging.info(f"Saved labeled clusters to {output_dir}/labeled_clusters.json")
        
        return labeled_clusters