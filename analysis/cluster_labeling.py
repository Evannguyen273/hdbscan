"""
Enhanced Cluster Labeling Module for HDBSCAN Pipeline
LLM-based intelligent cluster naming with versioned storage support
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from datetime import datetime
import openai
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
import os

logger = logging.getLogger(__name__)

class EnhancedClusterLabeler:
    """
    Enhanced cluster labeling system with LLM integration and versioned storage support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cluster labeler with Azure OpenAI configuration
        
        Args:
            config: Configuration dictionary with OpenAI settings
        """
        self.config = config
        self.client = self._initialize_openai_client()
        self.label_cache = {}
        
    def _initialize_openai_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client"""
        try:
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            logger.info("Azure OpenAI client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def generate_cluster_labels(
        self,
        incidents_df: pd.DataFrame,
        cluster_labels: np.ndarray,
        tech_center: str,
        model_version: str,
        max_samples_per_cluster: int = 20
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate intelligent labels for clusters using LLM analysis
        
        Args:
            incidents_df: DataFrame with incident data including summaries
            cluster_labels: Array of cluster assignments
            tech_center: Technology center name
            model_version: Model version (e.g., "2025_q2")
            max_samples_per_cluster: Maximum incidents to analyze per cluster
            
        Returns:
            Dictionary mapping cluster IDs to label information
        """
        logger.info(f"Generating cluster labels for {tech_center} - {model_version}")
        
        cluster_info = {}
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        
        for cluster_id in unique_clusters:
            logger.debug(f"Processing cluster {cluster_id}")
            
            # Get incidents for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_incidents = incidents_df[cluster_mask]
            
            if len(cluster_incidents) == 0:
                continue
                
            # Sample incidents for analysis
            sample_incidents = cluster_incidents.sample(
                n=min(max_samples_per_cluster, len(cluster_incidents)),
                random_state=42
            )
            
            # Generate label using LLM
            label_info = self._generate_label_for_cluster(
                sample_incidents,
                cluster_id,
                tech_center
            )
            
            # Add metadata
            label_info.update({
                "cluster_size": len(cluster_incidents),
                "sample_size": len(sample_incidents),
                "tech_center": tech_center,
                "model_version": model_version,
                "generated_at": datetime.now().isoformat()
            })
            
            cluster_info[cluster_id] = label_info
            
        logger.info(f"Generated labels for {len(cluster_info)} clusters")
        return cluster_info
    
    def _generate_label_for_cluster(
        self,
        incidents: pd.DataFrame,
        cluster_id: int,
        tech_center: str
    ) -> Dict[str, Any]:
        """
        Generate a label for a specific cluster using LLM
        
        Args:
            incidents: DataFrame of incidents in the cluster
            cluster_id: Cluster identifier
            tech_center: Technology center name
            
        Returns:
            Dictionary with label information
        """
        # Prepare incident summaries for analysis
        summaries = incidents['combined_incidents_summary'].tolist()
        
        # Create cache key
        cache_key = f"{tech_center}_{cluster_id}_{hash(str(sorted(summaries)))}"
        
        if cache_key in self.label_cache:
            logger.debug(f"Using cached label for cluster {cluster_id}")
            return self.label_cache[cache_key]
        
        # Prepare prompt for LLM
        prompt = self._create_labeling_prompt(summaries, tech_center)
        
        try:
            # Call Azure OpenAI for label generation
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert IT analyst specializing in incident classification and pattern recognition."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500
            )
            
            # Parse LLM response
            label_info = self._parse_llm_response(response.choices[0].message.content)
            
            # Cache the result
            self.label_cache[cache_key] = label_info
            
            logger.debug(f"Generated label for cluster {cluster_id}: {label_info.get('primary_label', 'Unknown')}")
            return label_info
            
        except Exception as e:
            logger.error(f"Error generating label for cluster {cluster_id}: {e}")
            return self._create_fallback_label(incidents, cluster_id)
    
    def _create_labeling_prompt(self, summaries: List[str], tech_center: str) -> str:
        """
        Create a structured prompt for LLM-based cluster labeling
        
        Args:
            summaries: List of incident summaries
            tech_center: Technology center name
            
        Returns:
            Formatted prompt string
        """
        # Limit summaries for prompt size management
        sample_summaries = summaries[:15]  # Use first 15 summaries
        
        prompt = f"""
Analyze the following IT incident summaries from {tech_center} and provide a comprehensive cluster label.

INCIDENT SUMMARIES:
{chr(10).join([f"{i+1}. {summary}" for i, summary in enumerate(sample_summaries)])}

Please provide your analysis in the following JSON format:
{{
    "primary_label": "Primary descriptive name for this cluster (2-4 words)",
    "detailed_description": "Detailed description of the common issue pattern",
    "issue_category": "Main category (e.g., Network, Database, Authentication, Performance)",
    "severity_pattern": "Common severity level or pattern",
    "common_keywords": ["keyword1", "keyword2", "keyword3"],
    "root_cause_pattern": "Common root cause or technical pattern",
    "confidence_score": 0.85
}}

Focus on:
1. The most common technical issue or pattern
2. The primary affected system or component
3. The type of problem (outage, performance, error, etc.)
4. Make labels concise but descriptive
5. Confidence score from 0-1 based on pattern clarity
"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured label information
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed label information
        """
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                label_info = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["primary_label", "detailed_description", "issue_category"]
                for field in required_fields:
                    if field not in label_info:
                        logger.warning(f"Missing required field in LLM response: {field}")
                        label_info[field] = "Unknown"
                
                return label_info
            else:
                logger.warning("No valid JSON found in LLM response")
                return self._create_default_label(response_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM JSON response: {e}")
            return self._create_default_label(response_text)
    
    def _create_default_label(self, response_text: str) -> Dict[str, Any]:
        """Create default label from unparseable response"""
        return {
            "primary_label": "Mixed Issues",
            "detailed_description": response_text[:200] + "..." if len(response_text) > 200 else response_text,
            "issue_category": "General",
            "severity_pattern": "Mixed",
            "common_keywords": [],
            "root_cause_pattern": "Various",
            "confidence_score": 0.3
        }
    
    def _create_fallback_label(self, incidents: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """
        Create fallback label using keyword analysis when LLM fails
        
        Args:
            incidents: DataFrame of incidents in cluster
            cluster_id: Cluster identifier
            
        Returns:
            Fallback label information
        """
        # Simple keyword-based labeling
        all_text = ' '.join(incidents['combined_incidents_summary'].fillna('').astype(str))
        words = all_text.lower().split()
        
        # Common IT keywords to look for
        keyword_categories = {
            "network": ["network", "connectivity", "connection", "ping", "dns"],
            "database": ["database", "sql", "query", "db", "oracle", "mysql"],
            "authentication": ["login", "password", "auth", "access", "permission"],
            "performance": ["slow", "performance", "timeout", "latency", "response"],
            "server": ["server", "host", "vm", "machine", "instance"],
            "application": ["application", "app", "software", "service", "api"]
        }
        
        # Count keyword occurrences
        category_scores = {}
        for category, keywords in keyword_categories.items():
            score = sum(words.count(keyword) for keyword in keywords)
            if score > 0:
                category_scores[category] = score
        
        # Determine primary category
        if category_scores:
            primary_category = max(category_scores, key=category_scores.get)
            primary_label = f"{primary_category.title()} Issues"
        else:
            primary_label = f"Cluster {cluster_id}"
        
        return {
            "primary_label": primary_label,
            "detailed_description": f"Cluster containing {len(incidents)} incidents with common patterns",
            "issue_category": primary_category if category_scores else "General",
            "severity_pattern": "Mixed",
            "common_keywords": list(category_scores.keys()) if category_scores else [],
            "root_cause_pattern": "Analysis required",
            "confidence_score": 0.5 if category_scores else 0.2,
            "fallback_method": "keyword_analysis"
        }
    
    def generate_domain_groupings(
        self,
        cluster_labels_info: Dict[int, Dict[str, Any]],
        max_domains: int = 20
    ) -> Dict[str, List[int]]:
        """
        Group clusters into higher-level domains based on categories
        
        Args:
            cluster_labels_info: Dictionary of cluster label information
            max_domains: Maximum number of domains to create
            
        Returns:
            Dictionary mapping domain names to cluster IDs
        """
        logger.info(f"Generating domain groupings for {len(cluster_labels_info)} clusters")
        
        # Group by issue category
        category_clusters = {}
        for cluster_id, info in cluster_labels_info.items():
            category = info.get("issue_category", "General")
            if category not in category_clusters:
                category_clusters[category] = []
            category_clusters[category].append(cluster_id)
        
        # If we have more categories than max_domains, merge smaller ones
        if len(category_clusters) > max_domains:
            # Sort by cluster count and merge smallest categories into "Other"
            sorted_categories = sorted(category_clusters.items(), key=lambda x: len(x[1]), reverse=True)
            
            main_categories = dict(sorted_categories[:max_domains-1])
            other_clusters = []
            for _, clusters in sorted_categories[max_domains-1:]:
                other_clusters.extend(clusters)
            
            if other_clusters:
                main_categories["Other"] = other_clusters
                
            category_clusters = main_categories
        
        logger.info(f"Created {len(category_clusters)} domain groupings")
        return category_clusters
    
    def save_labels_to_json(
        self,
        cluster_labels_info: Dict[int, Dict[str, Any]],
        filepath: str,
        include_metadata: bool = True
    ) -> None:
        """
        Save cluster labels to JSON file for versioned storage
        
        Args:
            cluster_labels_info: Dictionary of cluster label information
            filepath: Output file path
            include_metadata: Whether to include generation metadata
        """
        try:
            output_data = {
                "cluster_labels": cluster_labels_info,
                "generation_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_clusters": len(cluster_labels_info),
                    "labeling_method": "llm_enhanced",
                    "version": "2.0"
                } if include_metadata else {}
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Cluster labels saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving cluster labels: {e}")
    
    def load_labels_from_json(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        """
        Load cluster labels from JSON file
        
        Args:
            filepath: Input file path
            
        Returns:
            Dictionary of cluster label information
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both new and legacy formats
            if "cluster_labels" in data:
                cluster_labels = data["cluster_labels"]
            else:
                cluster_labels = data
                
            # Convert string keys to integers
            cluster_labels = {int(k): v for k, v in cluster_labels.items()}
            
            logger.info(f"Loaded labels for {len(cluster_labels)} clusters from {filepath}")
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Error loading cluster labels: {e}")
            return {}
    
    def get_label_summary(self, cluster_labels_info: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for cluster labels
        
        Args:
            cluster_labels_info: Dictionary of cluster label information
            
        Returns:
            Summary statistics
        """
        if not cluster_labels_info:
            return {}
        
        categories = [info.get("issue_category", "Unknown") for info in cluster_labels_info.values()]
        confidence_scores = [info.get("confidence_score", 0) for info in cluster_labels_info.values()]
        
        summary = {
            "total_clusters": len(cluster_labels_info),
            "category_distribution": dict(Counter(categories)),
            "average_confidence": np.mean(confidence_scores) if confidence_scores else 0,
            "high_confidence_clusters": sum(1 for score in confidence_scores if score >= 0.8),
            "low_confidence_clusters": sum(1 for score in confidence_scores if score < 0.5)
        }
        
        return summary

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