"""
Enhanced Cluster Labeling and Domain Grouping Module
LLM-based intelligent cluster naming with domain grouping support
Based on the full_script.py workflow
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from openai import AzureOpenAI
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

from config.config import load_config

class EnhancedClusterLabeler:
    """
    Enhanced cluster labeling using Azure OpenAI for semantic analysis
    Includes domain grouping functionality from full_script.py
    """
    
    def __init__(self, config=None):
        """Initialize labeler with configuration-driven setup"""
        self.config = config if config is not None else load_config()
        self.client = self._initialize_openai_client()
        self.logger = logging.getLogger(__name__)
        
        # Labeling parameters from config
        labeling_config = self.config.get('clustering.labeling', {})
        self.max_examples = labeling_config.get('max_examples_per_cluster', 10)
        self.model_name = labeling_config.get('model_name', 'gpt-4o-mini')
        self.temperature = labeling_config.get('temperature', 0.1)
        
        # Domain grouping parameters
        self.max_domains = labeling_config.get('max_domains', 20)
        self.min_clusters_per_domain = labeling_config.get('min_clusters_per_domain', 2)
        
    def _initialize_openai_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client using configuration"""
        try:
            # Use environment variables for Azure OpenAI
            import os
            client = AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.logger.info("Azure OpenAI client initialized successfully")
            return client
            
        except Exception as e:
            self.logger.error("Failed to initialize OpenAI client: %s", str(e))
            raise
    
    def generate_cluster_labels(self, incidents_df: pd.DataFrame, cluster_labels: np.ndarray, 
                              reduced_embeddings: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Generate semantic labels for clusters using LLM
        
        Args:
            incidents_df: DataFrame with incident data including 'combined_incidents_summary'
            cluster_labels: Array of cluster assignments for each incident
            reduced_embeddings: UMAP reduced embeddings for spatial analysis
            
        Returns:
            Dictionary mapping cluster_id to cluster information
        """
        cluster_info = {}
        
        # Get unique clusters (excluding noise cluster -1)
        unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
        self.logger.info(f"Generating labels for {len(unique_clusters)} clusters")
        
        for cluster_id in unique_clusters:
            try:
                # Get incidents for this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_incidents = incidents_df[cluster_mask]['combined_incidents_summary'].tolist()
                cluster_coords = reduced_embeddings[cluster_mask]
                
                # Sample incidents for labeling (max 10 for LLM)
                sample_incidents = cluster_incidents[:self.max_examples]
                
                # Calculate cluster center and spread
                center_x, center_y = np.mean(cluster_coords, axis=0)
                spread = np.std(cluster_coords, axis=0)
                
                # Generate label using LLM
                label_result = self._generate_label_for_cluster(sample_incidents)
                
                cluster_info[cluster_id] = {
                    'primary_label': label_result.get('primary_label', f'Cluster_{cluster_id}'),
                    'description': label_result.get('description', ''),
                    'keywords': label_result.get('keywords', []),
                    'domain': label_result.get('domain', 'Unknown'),
                    'size': len(cluster_incidents),
                    'center_x': float(center_x),
                    'center_y': float(center_y),
                    'spread_x': float(spread[0]),
                    'spread_y': float(spread[1]),
                    'sample_incidents': sample_incidents[:5],  # Keep top 5 for reference
                    'incident_numbers': incidents_df[cluster_mask]['number'].tolist()  # Store incident numbers
                }
                
                self.logger.info("Generated label for cluster %d (%d incidents): %s", 
                               cluster_id, len(cluster_incidents), cluster_info[cluster_id]['primary_label'])
                
            except Exception as e:
                self.logger.error("Failed to generate label for cluster %d: %s", cluster_id, str(e))
                cluster_info[cluster_id] = {
                    'primary_label': f'Cluster_{cluster_id}',
                    'description': 'Auto-generated cluster',
                    'keywords': [],
                    'domain': 'Unknown',
                    'size': len(cluster_incidents) if 'cluster_incidents' in locals() else 0,
                    'center_x': 0.0,
                    'center_y': 0.0,
                    'spread_x': 0.0,
                    'spread_y': 0.0,
                    'sample_incidents': [],
                    'incident_numbers': []
                }
        
        return cluster_info
    
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
            4. The IT domain (e.g., Network, Database, Application, Infrastructure, Security, Storage, Authentication)

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
    
    def group_clusters_into_domains(self, cluster_info: Dict[int, Dict[str, Any]], 
                                  reduced_embeddings: np.ndarray, 
                                  cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Group clusters into domains based on spatial proximity and semantic similarity
        Implementation from full_script.py
        
        Args:
            cluster_info: Dictionary of cluster information from generate_cluster_labels
            reduced_embeddings: UMAP reduced embeddings
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary containing domain groupings and metadata
        """
        try:
            self.logger.info("Grouping clusters into domains...")
            
            # Extract cluster centers for domain grouping
            cluster_centers = []
            cluster_ids = []
            
            for cluster_id, info in cluster_info.items():
                cluster_centers.append([info['center_x'], info['center_y']])
                cluster_ids.append(cluster_id)
            
            cluster_centers = np.array(cluster_centers)
            
            # Determine optimal number of domains (max 20, min 3)
            n_clusters = len(cluster_ids)
            n_domains = min(self.max_domains, max(3, n_clusters // 3))
            
            # Use KMeans to group clusters into domains based on spatial proximity
            domain_kmeans = KMeans(n_clusters=n_domains, random_state=42, n_init=10)
            domain_assignments = domain_kmeans.fit_predict(cluster_centers)
            
            # Create domain groupings
            domains = {}
            for i, cluster_id in enumerate(cluster_ids):
                domain_id = int(domain_assignments[i])
                
                if domain_id not in domains:
                    domains[domain_id] = {
                        'clusters': [],
                        'total_incidents': 0,
                        'domain_center': domain_kmeans.cluster_centers_[domain_id].tolist(),
                        'cluster_labels': [],
                        'cluster_descriptions': [],
                        'cluster_keywords': []
                    }
                
                domains[domain_id]['clusters'].append(cluster_id)
                domains[domain_id]['total_incidents'] += cluster_info[cluster_id]['size']
                domains[domain_id]['cluster_labels'].append(cluster_info[cluster_id]['primary_label'])
                domains[domain_id]['cluster_descriptions'].append(cluster_info[cluster_id]['description'])
                domains[domain_id]['cluster_keywords'].extend(cluster_info[cluster_id]['keywords'])
            
            # Generate domain names using LLM
            for domain_id, domain_data in domains.items():
                domain_name, domain_description = self._generate_domain_label(domain_data)
                domains[domain_id]['domain_name'] = domain_name
                domains[domain_id]['domain_description'] = domain_description
                
                self.logger.info(f"Domain {domain_id}: {domain_name} ({domain_data['total_incidents']} incidents)")
            
            # Create mapping from cluster_id to domain
            cluster_to_domain = {}
            for domain_id, domain_data in domains.items():
                for cluster_id in domain_data['clusters']:
                    cluster_to_domain[cluster_id] = {
                        'domain_id': domain_id,
                        'domain_name': domain_data['domain_name']
                    }
            
            return {
                'domains': domains,
                'cluster_to_domain': cluster_to_domain,
                'n_domains': n_domains,
                'domain_centers': domain_kmeans.cluster_centers_.tolist()
            }
            
        except Exception as e:
            self.logger.error("Failed to group clusters into domains: %s", str(e))
            return {
                'domains': {},
                'cluster_to_domain': {},
                'n_domains': 0,
                'domain_centers': []
            }
    
    def _generate_domain_label(self, domain_data: Dict[str, Any]) -> Tuple[str, str]:
        """Generate domain name and description using LLM"""
        try:
            # Prepare cluster information for domain naming
            cluster_labels = domain_data['cluster_labels']
            cluster_descriptions = domain_data['cluster_descriptions']
            total_incidents = domain_data['total_incidents']
            
            cluster_info_text = "\n".join([
                f"- {label}: {desc}" for label, desc in zip(cluster_labels, cluster_descriptions)
            ])
            
            prompt = f"""
            Analyze the following cluster groups and create a high-level domain name and description:

            Clusters in this domain ({total_incidents} total incidents):
            {cluster_info_text}

            Please provide:
            1. A concise domain name (2-3 words) that captures the overarching category
            2. A brief description (1-2 sentences) of what this domain represents

            Focus on the common theme across all clusters. This should be a high-level IT domain category.

            Respond in JSON format:
            {{
                "domain_name": "...",
                "domain_description": "..."
            }}
            """
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=300
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            result = json.loads(response_text)
            return result.get('domain_name', 'Unknown Domain'), result.get('domain_description', '')
            
        except Exception as e:
            self.logger.error("Failed to generate domain label: %s", str(e))
            return 'Unknown Domain', 'Unable to categorize domain'
    
    def create_labeled_dataset(self, incidents_df: pd.DataFrame, cluster_labels: np.ndarray,
                             cluster_info: Dict[int, Dict[str, Any]], 
                             domain_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Create final labeled dataset with cluster and domain information
        
        Args:
            incidents_df: Original incidents dataframe
            cluster_labels: Cluster assignments
            cluster_info: Cluster information from generate_cluster_labels
            domain_info: Domain information from group_clusters_into_domains
            
        Returns:
            DataFrame with cluster and domain labels added
        """
        try:
            # Create copy of incidents dataframe
            labeled_df = incidents_df.copy()
            
            # Add cluster assignments
            labeled_df['cluster_id'] = cluster_labels
            
            # Add cluster labels
            labeled_df['cluster_label'] = labeled_df['cluster_id'].map(
                lambda x: cluster_info.get(x, {}).get('primary_label', 'Noise') if x != -1 else 'Noise'
            )
            
            # Add cluster descriptions
            labeled_df['cluster_description'] = labeled_df['cluster_id'].map(
                lambda x: cluster_info.get(x, {}).get('description', '') if x != -1 else ''
            )
            
            # Add domain information
            cluster_to_domain = domain_info.get('cluster_to_domain', {})
            labeled_df['domain_id'] = labeled_df['cluster_id'].map(
                lambda x: cluster_to_domain.get(x, {}).get('domain_id', -1) if x != -1 else -1
            )
            
            labeled_df['domain_name'] = labeled_df['cluster_id'].map(
                lambda x: cluster_to_domain.get(x, {}).get('domain_name', 'Noise') if x != -1 else 'Noise'
            )
            
            self.logger.info(f"Created labeled dataset with {len(labeled_df)} incidents")
            
            return labeled_df
            
        except Exception as e:
            self.logger.error("Failed to create labeled dataset: %s", str(e))
            return incidents_df
    
    def get_analysis_summary(self, cluster_info: Dict[int, Dict[str, Any]], 
                           domain_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics of the clustering and domain analysis"""
        try:
            # Cluster statistics
            total_clusters = len(cluster_info)
            total_incidents = sum(info['size'] for info in cluster_info.values())
            avg_cluster_size = total_incidents / total_clusters if total_clusters > 0 else 0
            
            cluster_sizes = [info['size'] for info in cluster_info.values()]
            
            # Domain statistics
            domains = domain_info.get('domains', {})
            total_domains = len(domains)
            
            domain_sizes = [domain['total_incidents'] for domain in domains.values()]
            avg_domain_size = sum(domain_sizes) / total_domains if total_domains > 0 else 0
            
            summary = {
                'total_clusters': total_clusters,
                'total_domains': total_domains,
                'total_incidents': total_incidents,
                'avg_cluster_size': round(avg_cluster_size, 2),
                'avg_domain_size': round(avg_domain_size, 2),
                'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                'smallest_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
                'largest_domain_size': max(domain_sizes) if domain_sizes else 0,
                'smallest_domain_size': min(domain_sizes) if domain_sizes else 0,
                'domain_names': [domain['domain_name'] for domain in domains.values()]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error("Failed to generate analysis summary: %s", str(e))
            return {}

    def run_complete_analysis(self, incidents_df: pd.DataFrame, cluster_labels: np.ndarray,
                            reduced_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Run complete cluster labeling and domain grouping analysis
        
        Args:
            incidents_df: DataFrame with incident data
            cluster_labels: Cluster assignments from HDBSCAN
            reduced_embeddings: UMAP reduced embeddings
            
        Returns:
            Complete analysis results
        """
        try:
            self.logger.info("🏷️ Starting complete cluster and domain analysis...")
            
            # Step 1: Generate cluster labels
            self.logger.info("📋 Step 1: Generating cluster labels...")
            cluster_info = self.generate_cluster_labels(incidents_df, cluster_labels, reduced_embeddings)
            
            # Step 2: Group clusters into domains
            self.logger.info("🗂️ Step 2: Grouping clusters into domains...")
            domain_info = self.group_clusters_into_domains(cluster_info, reduced_embeddings, cluster_labels)
            
            # Step 3: Create labeled dataset
            self.logger.info("📊 Step 3: Creating labeled dataset...")
            labeled_df = self.create_labeled_dataset(incidents_df, cluster_labels, cluster_info, domain_info)
            
            # Step 4: Generate summary
            self.logger.info("📈 Step 4: Generating analysis summary...")
            summary = self.get_analysis_summary(cluster_info, domain_info)
            
            results = {
                'cluster_info': cluster_info,
                'domain_info': domain_info,
                'labeled_dataframe': labeled_df,
                'summary': summary
            }
            
            self.logger.info("✅ Complete analysis finished successfully!")
            self.logger.info(f"📊 Results: {summary.get('total_clusters', 0)} clusters grouped into {summary.get('total_domains', 0)} domains")
            
            return results
            
        except Exception as e:
            self.logger.error("❌ Complete analysis failed: %s", str(e))
            raise