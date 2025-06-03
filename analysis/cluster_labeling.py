"""
Enhanced Cluster Labeling Module for HDBSCAN Pipeline
LLM-based intelligent cluster naming with versioned storage support
"""

import logging
import json
from typing import Dict, List, Any, Optional
from openai import AzureOpenAI
import pandas as pd

from config.config import load_config

class EnhancedClusterLabeler:
    """
    Enhanced cluster labeling using Azure OpenAI for semantic analysis
    """
    
    def __init__(self, config=None):
        """Initialize labeler with configuration-driven setup"""
        self.config = config if config is not None else load_config()
        self.client = self._initialize_openai_client()
        self.logger = logging.getLogger(__name__)
        
        # Labeling parameters from config
        labeling_config = self.config.get('clustering.labeling', {})
        self.max_examples = labeling_config.get('max_examples_per_cluster', 10)
        self.model_name = labeling_config.get('model_name', 'gpt-4')
        self.temperature = labeling_config.get('temperature', 0.1)
        
    def _initialize_openai_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client using configuration"""
        try:
            openai_config = self.config.azure.openai
            client = AzureOpenAI(
                api_key=openai_config.api_key,
                api_version=openai_config.api_version,
                azure_endpoint=openai_config.endpoint
            )
            self.logger.info("Azure OpenAI client initialized successfully")
            return client
            
        except Exception as e:
            self.logger.error("Failed to initialize OpenAI client: %s", str(e))
            raise
    
    def generate_cluster_labels(self, cluster_data: Dict[int, List[str]]) -> Dict[int, Dict[str, str]]:
        """Generate semantic labels for clusters"""
        cluster_labels = {}
        
        for cluster_id, incidents in cluster_data.items():
            try:
                # Sample incidents for labeling
                sample_incidents = incidents[:self.max_examples]
                
                # Generate label using LLM
                label_result = self._generate_label_for_cluster(sample_incidents)
                
                cluster_labels[cluster_id] = {
                    'primary_label': label_result.get('primary_label', f'Cluster_{cluster_id}'),
                    'description': label_result.get('description', ''),
                    'keywords': label_result.get('keywords', []),
                    'domain': label_result.get('domain', 'Unknown')
                }
                
                self.logger.info("Generated label for cluster %d: %s", 
                               cluster_id, cluster_labels[cluster_id]['primary_label'])
                
            except Exception as e:
                self.logger.error("Failed to generate label for cluster %d: %s", cluster_id, str(e))
                cluster_labels[cluster_id] = {
                    'primary_label': f'Cluster_{cluster_id}',
                    'description': 'Auto-generated cluster',
                    'keywords': [],
                    'domain': 'Unknown'
                }
        
        return cluster_labels
    
    def _generate_label_for_cluster(self, sample_incidents: List[str]) -> Dict[str, Any]:
        """Generate label for a single cluster using LLM"""
        try:
            # Prepare prompt
            incidents_text = "\n".join([f"- {incident}" for incident in sample_incidents])
            
            prompt = f"""
            Analyze the following IT incidents and provide a semantic label and categorization:

            Incidents:
            {incidents_text}

            Please provide:
            1. A concise primary label (2-4 words) that captures the main issue type
            2. A brief description (1-2 sentences) of what this cluster represents
            3. 3-5 keywords that characterize these incidents
            4. The IT domain (e.g., Network, Database, Application, Infrastructure, Security)

            Respond in JSON format:
            {{
                "primary_label": "...",
                "description": "...",
                "keywords": [...],
                "domain": "..."
            }}
            """
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            result = json.loads(response_text)
            return result
            
        except Exception as e:
            self.logger.error("Failed to generate LLM label: %s", str(e))
            return {
                'primary_label': 'Unknown_Issue',
                'description': 'Unable to categorize',
                'keywords': [],
                'domain': 'Unknown'
            }
    
    def validate_and_refine_labels(self, cluster_labels: Dict[int, Dict[str, str]], 
                                 cluster_data: Dict[int, List[str]]) -> Dict[int, Dict[str, str]]:
        """Validate and refine generated labels"""
        refined_labels = {}
        
        for cluster_id, label_info in cluster_labels.items():
            try:
                # Basic validation
                if not label_info.get('primary_label') or label_info['primary_label'] == '':
                    label_info['primary_label'] = f'Cluster_{cluster_id}'
                
                # Ensure domain is valid
                valid_domains = ['Network', 'Database', 'Application', 'Infrastructure', 'Security', 'Unknown']
                if label_info.get('domain') not in valid_domains:
                    label_info['domain'] = 'Unknown'
                
                # Ensure keywords is a list
                if not isinstance(label_info.get('keywords'), list):
                    label_info['keywords'] = []
                
                refined_labels[cluster_id] = label_info
                
            except Exception as e:
                self.logger.error("Failed to validate label for cluster %d: %s", cluster_id, str(e))
                refined_labels[cluster_id] = cluster_labels[cluster_id]
        
        return refined_labels

# Example usage and testing
if __name__ == "__main__":
    from logging_setup import setup_detailed_logging
    setup_detailed_logging()
    
    # Test configuration
    config = {
        "openai": {
            "deployment_name": "gpt-4",
            "api_version": "2023-12-01-preview"
        }
    }
    
    # Mock data for testing
    incidents_df = pd.DataFrame({
        'combined_incidents_summary': [
            "Database connection timeout in production environment",
            "Unable to connect to SQL server, query performance issues",
            "Network connectivity problems affecting multiple users",
            "Authentication service returning 500 errors",
            "Server performance degradation, high CPU usage"
        ]
    })
    
    cluster_labels = np.array([0, 0, 1, 2, 1])
    
    try:
        labeler = EnhancedClusterLabeler(config)
        
        # Generate labels
        labels_info = labeler.generate_cluster_labels(
            incidents_df=incidents_df,
            cluster_labels=cluster_labels,
            tech_center="BT-TC-Data Analytics",
            model_version="2025_q2"
        )
        
        print("Generated Labels:")
        for cluster_id, info in labels_info.items():
            print(f"Cluster {cluster_id}: {info['primary_label']}")
            
        # Generate domain groupings
        domains = labeler.generate_domain_groupings(labels_info)
        print(f"\nDomain Groupings: {domains}")
        
        # Get summary
        summary = labeler.get_label_summary(labels_info)
        print(f"\nLabel Summary: {summary}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print("Note: This test requires Azure OpenAI credentials to be configured")