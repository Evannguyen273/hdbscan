#!/usr/bin/env python3
"""
Enhanced Training Pipeline with Hybrid Domain Grouping
Integrates the spatial + semantic domain grouping approach from Untitled-1
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# ML imports
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from openai import AzureOpenAI

# Local imports
from config.config import get_config
from logging_setup import setup_detailed_logging


class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline with integrated hybrid domain grouping
    Combines spatial proximity (UMAP) with semantic understanding (LLM)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced training pipeline"""
        # Setup logging
        setup_detailed_logging(logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = get_config(config_path)
        
        # Initialize Azure OpenAI client
        self._setup_openai_client()
        
        # Create output directories
        self.results_path = Path("results")
        self.results_path.mkdir(exist_ok=True)
        
        self.logger.info("Enhanced Training Pipeline initialized with hybrid domain grouping")
    
    def _setup_openai_client(self):
        """Setup Azure OpenAI client for domain naming"""
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=self.config.azure.openai.endpoint,
                api_key=self.config.azure.openai.api_key,
                api_version=self.config.azure.openai.api_version,
            )
            self.deployment_name = self.config.azure.openai.deployment_name
            self.logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def run_training_with_domains(self, tech_center: str, quarter: str, year: int = 2024):
        """
        Run complete training pipeline with hybrid domain grouping
        
        Args:
            tech_center: Name of the tech center
            quarter: Quarter (q1, q2, q3, q4)
            year: Year for training data
            
        Returns:
            Dictionary with training results including domains
        """
        try:
            self.logger.info(f"Starting enhanced training for {tech_center} - Q{quarter} {year}")
            
            # 1. Preprocess and generate embeddings
            # (This would integrate with your existing preprocessing pipeline)
            embeddings_data = self._preprocess_and_embed(tech_center, quarter, year)
            
            # 2. Perform HDBSCAN clustering
            clustering_results = self._perform_clustering(embeddings_data, tech_center)
            
            # 3. Generate cluster labels using LLM
            labeled_clusters = self._generate_cluster_labels(clustering_results, tech_center)
            
            # 4. Apply hybrid domain grouping (core feature from Untitled-1)
            domain_results = self._group_clusters_into_domains(
                labeled_clusters=labeled_clusters['labeled_clusters'],
                clusters_info=labeled_clusters['clusters_info'],
                umap_embeddings=clustering_results['umap_embeddings'],
                cluster_labels=clustering_results['cluster_labels'],
                tech_center=tech_center,
                quarter=quarter,
                year=year
            )
            
            # 5. Save comprehensive results
            final_results = self._save_training_results(
                domain_results, clustering_results, tech_center, quarter, year
            )
            
            self.logger.info(f"Enhanced training completed for {tech_center}")
            self._print_training_summary(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Enhanced training failed for {tech_center}: {e}")
            raise
    
    def _group_clusters_into_domains(self, labeled_clusters: Dict, clusters_info: Dict, 
                                   umap_embeddings: np.ndarray, cluster_labels: np.ndarray,
                                   tech_center: str, quarter: str, year: int,
                                   max_domains: int = 20) -> Dict:
        """
        Group clusters into domains using hybrid approach from Untitled-1
        Combines coordinate proximity with semantic similarity
        """
        self.logger.info(f"Grouping clusters into domains with auto-optimization (max: {max_domains} domains)")
        
        # Skip noise cluster (-1) for domain grouping
        clusters_to_group = {k: v for k, v in labeled_clusters.items() if k != "-1"}
        
        # If very few clusters, create simple domain structure
        if len(clusters_to_group) <= 3:
            return self._create_simple_domain_structure(clusters_to_group, tech_center)
        
        # 1. Calculate cluster centroids in UMAP space
        unique_clusters = np.unique(cluster_labels)
        cluster_centroids = {}
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise
                continue
            cluster_mask = cluster_labels == cluster_id
            cluster_points = umap_embeddings[cluster_mask]
            if len(cluster_points) > 0:
                cluster_centroids[cluster_id] = np.mean(cluster_points, axis=0)

        # 2. Convert centroids to array for hierarchical clustering
        centroid_ids = list(cluster_centroids.keys())
        centroid_coords = np.array([cluster_centroids[cid] for cid in centroid_ids])

        # 3. Determine optimal number of domains using internal metrics
        optimal_domains = self._determine_optimal_domains(centroid_coords, max_domains)
        
        # 4. Apply hierarchical clustering with optimal domain count
        hc = AgglomerativeClustering(n_clusters=optimal_domains, linkage='ward')
        domain_labels = hc.fit_predict(centroid_coords)

        # 5. Create initial domain mapping
        initial_domains = {}
        for i, domain_id in enumerate(domain_labels):
            if domain_id not in initial_domains:
                initial_domains[domain_id] = []
            initial_domains[domain_id].append(centroid_ids[i])

        # 6. Remove empty domains
        initial_domains = {k: v for k, v in initial_domains.items() if v}
        domain_count = len(initial_domains)
        
        self.logger.info(f"Initial domain grouping created {domain_count} domains based on cluster coordinates")

        # 7. Use LLM to name and refine domains
        final_domains = []
        all_standardized_labels = {}

        # Process each domain with LLM
        for domain_id, cluster_ids in initial_domains.items():
            domain_result = self._process_domain_with_llm(
                domain_id, cluster_ids, labeled_clusters, clusters_info, domain_count
            )
            
            if domain_result and cluster_ids:
                final_domains.append({
                    "domain_name": domain_result["domain_name"],
                    "description": domain_result["description"],
                    "clusters": [self._safe_cluster_id_to_int(cid) for cid in cluster_ids]
                })
                
                if "standardized_labels" in domain_result:
                    for cluster_id, new_label in domain_result["standardized_labels"].items():
                        clean_cluster_id = str(self._safe_cluster_id_to_int(cluster_id))
                        all_standardized_labels[clean_cluster_id] = new_label

        # 8. Add noise domain
        final_domains.append({
            "domain_name": "Noise",
            "description": "Uncategorized incidents",
            "clusters": [-1]
        })

        # Apply standardized labels
        for cluster_id, new_label in all_standardized_labels.items():
            if cluster_id in labeled_clusters:
                labeled_clusters[cluster_id]["topic"] = new_label

        return {
            "domains": {"domains": final_domains},
            "labeled_clusters": labeled_clusters,
            "clusters_info": clusters_info,
            "tech_center": tech_center,
            "quarter": quarter,
            "year": year,
            "timestamp": datetime.now().isoformat()
        }
    
    def _determine_optimal_domains(self, centroid_coords: np.ndarray, max_domains: int) -> int:
        """Determine optimal number of domains using silhouette and CH scores"""
        max_domains_to_try = min(max_domains, len(centroid_coords) - 1)
        max_domains_to_try = max(2, max_domains_to_try)
        
        scores = []
        domain_counts = range(2, max_domains_to_try + 1)
        
        self.logger.info("Determining optimal number of domains...")
        for n_domains in domain_counts:
            hc = AgglomerativeClustering(n_clusters=n_domains, linkage='ward')
            domain_labels = hc.fit_predict(centroid_coords)
            
            if len(np.unique(domain_labels)) > 1:
                try:
                    sil_score = silhouette_score(centroid_coords, domain_labels)
                    ch_score = calinski_harabasz_score(centroid_coords, domain_labels)
                    combined_score = sil_score + (ch_score / (1000 + ch_score))
                    scores.append(combined_score)
                    self.logger.info(f"  {n_domains} domains: silhouette={sil_score:.3f}, CH={ch_score:.1f}, combined={combined_score:.3f}")
                except Exception as e:
                    self.logger.warning(f"Error calculating score for {n_domains} domains: {e}")
                    scores.append(-1)
            else:
                scores.append(-1)

        # Find optimal number of domains
        valid_scores = [(i, score) for i, score in enumerate(scores) if score > 0]
        if valid_scores:
            best_idx, best_score = max(valid_scores, key=lambda x: x[1])
            optimal_domains = domain_counts[best_idx]
            self.logger.info(f"Selected optimal domain count: {optimal_domains} (score: {best_score:.3f})")
        else:
            optimal_domains = min(15, max(2, len(centroid_coords) // 5))
            self.logger.warning(f"Could not determine optimal domains. Using default: {optimal_domains}")
        
        return optimal_domains
    
    def _process_domain_with_llm(self, domain_id: int, cluster_ids: List, 
                               labeled_clusters: Dict, clusters_info: Dict, 
                               domain_count: int) -> Optional[Dict]:
        """Process a single domain with LLM for naming and standardization"""
        try:
            # Create prompt for this domain
            prompt_text = f"I have a group of related IT incident clusters that form a domain.\n\n"
            prompt_text += "CLUSTERS IN THIS DOMAIN:\n\n"
            
            for cluster_id in cluster_ids:
                str_cluster_id = str(cluster_id)
                if str_cluster_id in clusters_info:
                    size = clusters_info[str_cluster_id]["size"]
                    percentage = clusters_info[str_cluster_id]["percentage"]
                else:
                    size = 0
                    percentage = 0
                    
                if str_cluster_id in labeled_clusters:
                    topic = labeled_clusters[str_cluster_id]["topic"]
                    description = labeled_clusters[str_cluster_id]["description"]
                else:
                    topic = f"Cluster {cluster_id}"
                    description = "No description available"
                
                prompt_text += f"\nCluster {cluster_id} (Size: {size}, {percentage}%):\n"
                prompt_text += f"Current Topic Label: {topic}\n"
                prompt_text += f"Description: {description}\n"
            
            prompt_text += f"""
Based on these related clusters, please:
1. Provide an appropriate name for this domain of incidents
2. Write a brief description of what unifies these incidents
3. Standardize the topic labels to use consistent terminology within this domain

YOU MUST RETURN A VALID JSON OBJECT with this exact structure:
{{
  "domain_name": "Descriptive Domain Name",
  "description": "Brief description of what unifies these incidents",
  "standardized_labels": {{
    "{cluster_ids[0]}": "Standardized topic label for cluster {cluster_ids[0]}",
    ... for all clusters in the domain
  }}
}}

IMPORTANT: For all cluster IDs in standardized_labels, use ONLY NUMERIC values without any prefixes.
Ensure standardized labels maintain specific system names where present but use consistent terminology.
"""

            # Call Azure OpenAI with retry logic
            correlation_id = str(uuid.uuid4())
            retry_attempts = 3
            
            for attempt in range(retry_attempts):
                try:
                    self.logger.info(f"Processing domain {domain_id+1}/{domain_count} with {len(cluster_ids)} clusters (attempt {attempt+1})")
                    
                    response = self.openai_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[
                            {"role": "system", "content": "You are an IT domain classification expert. Return properly structured JSON with no additional text. Use numeric IDs only without any prefixes like 'cluster_'."},
                            {"role": "user", "content": prompt_text}
                        ],
                        temperature=0.3,
                        response_format={"type": "json_object"},
                        timeout=30,
                        user=correlation_id
                    )
                    
                    result_text = response.choices[0].message.content
                    domain_result = json.loads(result_text)
                    
                    if "domain_name" in domain_result and "description" in domain_result and "standardized_labels" in domain_result:
                        return domain_result
                    else:
                        missing = []
                        if "domain_name" not in domain_result: missing.append("domain_name")
                        if "description" not in domain_result: missing.append("description") 
                        if "standardized_labels" not in domain_result: missing.append("standardized_labels")
                        raise ValueError(f"Response missing required keys: {missing}")
                        
                except Exception as e:
                    if attempt < retry_attempts - 1:
                        wait_time = min(30, 2 ** attempt * 2)
                        self.logger.warning(f"Domain processing attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Domain processing failed after all attempts: {e}")
                        return self._create_fallback_domain_result(domain_id, cluster_ids, labeled_clusters)
            
        except Exception as e:
            self.logger.error(f"Error processing domain {domain_id}: {e}")
            return None
    
    def _safe_cluster_id_to_int(self, cid) -> int:
        """Helper function to safely convert cluster IDs to integers"""
        try:
            return int(cid)
        except ValueError:
            if isinstance(cid, str) and 'cluster_' in cid:
                try:
                    return int(cid.split('_')[-1])
                except (ValueError, IndexError):
                    self.logger.warning(f"Could not convert cluster ID '{cid}' to integer. Using -99 as placeholder.")
                    return -99
            else:
                self.logger.warning(f"Could not convert cluster ID '{cid}' to integer. Using -99 as placeholder.")
                return -99
    
    def _create_fallback_domain_result(self, domain_id: int, cluster_ids: List, labeled_clusters: Dict) -> Dict:
        """Create fallback domain result when LLM processing fails"""
        if cluster_ids:
            sample_cluster_id = str(cluster_ids[0])
            sample_topic = "Unknown"
            try:
                clean_id = str(self._safe_cluster_id_to_int(sample_cluster_id))
                sample_topic = labeled_clusters.get(clean_id, {}).get("topic", f"Domain {domain_id}")
            except:
                pass
            
            return {
                "domain_name": f"Domain {domain_id}: {sample_topic}",
                "description": "Group of related incidents",
                "standardized_labels": {
                    str(self._safe_cluster_id_to_int(cid)): labeled_clusters.get(str(self._safe_cluster_id_to_int(cid)), {}).get("topic", f"Cluster {cid}")
                    for cid in cluster_ids
                }
            }
        else:
            return {
                "domain_name": f"Domain {domain_id}",
                "description": "Empty domain",
                "standardized_labels": {}
            }
    
    def _create_simple_domain_structure(self, clusters_to_group: Dict, tech_center: str) -> Dict:
        """Create simple domain structure for cases with few clusters"""
        domains = {
            "domains": [
                {
                    "domain_name": f"{tech_center} - All Issues",
                    "description": "All incident clusters for this tech center",
                    "clusters": [self._safe_cluster_id_to_int(k) for k in clusters_to_group.keys()]
                }
            ]
        }      
        domains["domains"].append({
            "domain_name": "Noise", 
            "description": "Uncategorized incidents", 
            "clusters": [-1]
        })
        
        return {
            "domains": domains,
            "labeled_clusters": clusters_to_group,
            "clusters_info": {},
            "tech_center": tech_center,
            "timestamp": datetime.now().isoformat()
        }
    
    def _preprocess_and_embed(self, tech_center: str, quarter: str, year: int) -> Dict:
        """Placeholder for preprocessing and embedding generation"""
        # This would integrate with your existing preprocessing pipeline
        self.logger.info(f"Preprocessing and embedding generation for {tech_center}")
        # Return mock data structure for now
        return {
            "embeddings": np.random.rand(1000, 1536),  # Mock embeddings
            "incidents": pd.DataFrame(),  # Mock incident data
            "tech_center": tech_center,
            "quarter": quarter,
            "year": year
        }
    
    def _perform_clustering(self, embeddings_data: Dict, tech_center: str) -> Dict:
        """Placeholder for HDBSCAN clustering"""
        # This would integrate with your existing clustering logic
        self.logger.info(f"Performing HDBSCAN clustering for {tech_center}")
        # Return mock clustering results for now
        return {
            "cluster_labels": np.random.randint(-1, 10, 1000),  # Mock cluster labels
            "umap_embeddings": np.random.rand(1000, 2),  # Mock UMAP embeddings
            "clusterer": None,  # Mock clusterer object
            "tech_center": tech_center
        }
    
    def _generate_cluster_labels(self, clustering_results: Dict, tech_center: str) -> Dict:
        """Placeholder for cluster label generation"""
        # This would integrate with your existing label generation logic
        self.logger.info(f"Generating cluster labels for {tech_center}")
        # Return mock labeled clusters for now
        return {
            "labeled_clusters": {
                "0": {"topic": "Network Connectivity", "description": "Network related issues"},
                "1": {"topic": "Server Hardware", "description": "Server hardware problems"},
                "-1": {"topic": "Noise", "description": "Uncategorized incidents"}
            },
            "clusters_info": {
                "0": {"size": 100, "percentage": 40.0},
                "1": {"size": 80, "percentage": 32.0},
                "-1": {"size": 70, "percentage": 28.0}
            }
        }
    
    def _save_training_results(self, domain_results: Dict, clustering_results: Dict, 
                             tech_center: str, quarter: str, year: int) -> Dict:
        """Save comprehensive training results"""
        # Create output directory
        output_dir = self.results_path / f"{tech_center}_{year}_q{quarter}"
        output_dir.mkdir(exist_ok=True)
        
        # Save domain results
        domain_path = output_dir / "domains.json"
        with open(domain_path, "w") as f:
            json.dump(domain_results["domains"], f, indent=2)
          # Save labeled clusters with standardized labels
        labeled_clusters_path = output_dir / "labeled_clusters_standardized.json"
        with open(labeled_clusters_path, "w") as f:
            json.dump(domain_results["labeled_clusters"], f, indent=2)
        
        # Save BigQuery results (without embeddings for cost optimization)
        bigquery_results = self._prepare_bigquery_results(
            domain_results, clustering_results, tech_center, quarter, year
        )
        
        bigquery_path = output_dir / f"{bigquery_results['table_name']}.json"
        with open(bigquery_path, "w") as f:
            json.dump(bigquery_results, f, indent=2)
        
        self.logger.info(f"Saved BigQuery results to {bigquery_path}")
        
        # Create summary with versioned table information
        summary = {
            "tech_center": tech_center,
            "quarter": quarter,
            "year": year,
            "timestamp": domain_results["timestamp"],
            "domains_count": len(domain_results["domains"]["domains"]),
            "clusters_count": len([d for d in domain_results["domains"]["domains"] if d["domain_name"] != "Noise"]),
            
            # Cumulative training metadata
            "training_approach": "cumulative_24_months",
            "training_window": {
                "start_date": bigquery_results["table_metadata"]["training_window_start"],
                "end_date": bigquery_results["table_metadata"]["training_window_end"],
                "duration_months": 24
            },
            
            # Versioned table information
            "bigquery_table": {
                "table_name": bigquery_results["table_name"],
                "model_version": bigquery_results["table_metadata"]["model_version"],
                "record_count": len(bigquery_results["records"]),
                "storage_optimization": "embeddings_excluded"
            },
            
            "output_files": {
                "domains": str(domain_path),
                "labeled_clusters": str(labeled_clusters_path),
                "bigquery_results": str(bigquery_path)
            }
        }
        
        # Save summary
        summary_path = output_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training results saved to {output_dir}")
        
        return {**domain_results, "summary": summary, "output_dir": str(output_dir)}
    
    def _print_training_summary(self, results: Dict):
        """Print a summary of the training results"""
        summary = results["summary"]
        domains = results["domains"]["domains"]
        
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY - {summary['tech_center']}")
        print(f"{'='*60}")
        print(f"Quarter: Q{summary['quarter']} {summary['year']}")
        print(f"Total Domains: {summary['domains_count']}")
        print(f"Total Clusters: {summary['clusters_count']}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"\nDomain Breakdown:")
        print(f"{'-'*40}")
        
        for domain in domains:
            if domain['domain_name'] != 'Noise':
                print(f"📁 {domain['domain_name']}")
                print(f"   Clusters: {len(domain['clusters'])}")
                print(f"   Description: {domain['description']}")
                print()
        
        print(f"📁 Output Directory: {results['output_dir']}")
        print(f"{'='*60}")
      def _prepare_bigquery_results(self, domain_results: Dict, clustering_results: Dict, 
                                tech_center: str, quarter: str, year: int) -> Dict:
        """
        Prepare results for BigQuery storage without embeddings (cost optimization)
        
        Creates versioned table with cumulative training approach:
        - Table name: clustering_predictions_{year}_{quarter}_{tech_center_hash}
        - Contains results from cumulative 24-month training window
        - No embeddings stored (cost optimization)
        """        # Create versioned table name for this training cycle
        tech_center_hash = abs(hash(tech_center)) % 1000  # Short hash for table naming
        table_version = f"{year}_{quarter}_{tech_center_hash:03d}"
        table_name = f"clustering_predictions_{table_version}"
        
        # Training window metadata (cumulative approach)
        training_start_date = self._calculate_training_start_date(year, quarter)
        training_end_date = f"{year}-{self._quarter_to_month(quarter)}-30"
        
        bigquery_results = {
            "table_name": table_name,
            "table_metadata": {
                "tech_center": tech_center,
                "training_window_start": training_start_date,
                "training_window_end": training_end_date,
                "training_approach": "cumulative_24_months",
                "model_version": f"{year}_{quarter}_v1",
                "created_timestamp": datetime.now().isoformat(),
                "domain_count": len(domain_results["domains"]["domains"]),
                "storage_optimization": "embeddings_excluded"
            },
            "records": []
        }
          # Get cluster-to-domain mapping
        cluster_to_domain = {}
        for domain in domain_results["domains"]["domains"]:
            domain_name = domain["domain_name"]
            domain_desc = domain["description"]
            for cluster_id in domain["clusters"]:
                cluster_to_domain[cluster_id] = {
                    "domain_name": domain_name,
                    "domain_description": domain_desc
                }
        
        # Mock incident data for demonstration
        # In real implementation, this would come from preprocessed_incidents table
        # with cumulative 24-month window
        mock_incidents = [
            {
                "number": f"INC{1000000 + i}",
                "sys_created_on": "2024-01-15T10:30:00Z",
                "combined_incidents_summary": f"Sample incident {i} summary",
                "business_service": "IT Services",
                "cluster_id": i % 5,  # Mock cluster assignment
                "umap_x": np.random.uniform(-10, 10),
                "umap_y": np.random.uniform(-10, 10)
            }
            for i in range(100)  # Mock 100 incidents (real: ~88k for 24-month window)
        ]
        
        for incident in mock_incidents:
            cluster_id = incident["cluster_id"]
            
            # Get cluster label
            cluster_label = domain_results["labeled_clusters"].get(
                str(cluster_id), {}
            ).get("topic", f"Cluster {cluster_id}")
            
            # Get domain information
            domain_info = cluster_to_domain.get(cluster_id, {
                "domain_name": "Uncategorized",
                "domain_description": "No domain assigned"
            })
            
            # Create BigQuery record without embeddings (cost optimization)
            bigquery_record = {
                # Incident identifiers
                "number": incident["number"],
                "sys_created_on": incident["sys_created_on"],
                
                # Incident content
                "combined_incidents_summary": incident["combined_incidents_summary"],
                "business_service": incident["business_service"],
                
                # Clustering results
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "cluster_description": domain_results["labeled_clusters"].get(
                    str(cluster_id), {}
                ).get("description", ""),
                
                # Domain grouping (NEW)
                "domain_id": hash(domain_info["domain_name"]) % 1000,  # Simple domain ID
                "domain_name": domain_info["domain_name"],
                "domain_description": domain_info["domain_description"],
                "domain_confidence": 0.85,  # Mock confidence score
                
                # Training metadata (cumulative approach)
                "tech_center": tech_center,
                "quarter": quarter,
                "year": year,
                "model_version": f"{year}_{quarter}_v1",
                "training_timestamp": datetime.now().isoformat(),
                "training_window_months": 24,  # Cumulative 24-month window
                
                # UMAP coordinates only (no full embeddings for cost savings)
                "umap_x": float(incident["umap_x"]),
                "umap_y": float(incident["umap_y"])
                
                # NOTE: embedding ARRAY<FLOAT64> removed to reduce storage costs
                # Embeddings remain in preprocessed_incidents table
            }
            
            bigquery_results["records"].append(bigquery_record)
        
        self.logger.info(f"Prepared {len(bigquery_results['records'])} records for table {table_name}")
        return bigquery_results

    def _calculate_training_start_date(self, year: int, quarter: str) -> str:
        """
        Calculate training window start date for cumulative 24-month approach
        
        Example: Training in Q2 2025 uses data from Q2 2023 to Q2 2025 (24 months)
        """
        start_year = year - 2  # Go back 24 months
        quarter_month_map = {"q1": "01", "q2": "04", "q3": "07", "q4": "10"}
        start_month = quarter_month_map.get(quarter, "01")
        return f"{start_year}-{start_month}-01"
    
    def _quarter_to_month(self, quarter: str) -> str:
        """Convert quarter to end month"""
        quarter_end_map = {"q1": "03", "q2": "06", "q3": "09", "q4": "12"}
        return quarter_end_map.get(quarter, "12")
    
def main():
    """Example usage of the enhanced training pipeline"""
    # Initialize pipeline
    pipeline = EnhancedTrainingPipeline()
    
    # Example tech centers and quarters
    tech_centers = [
        "BT-TC-Product Development & Engineering",
        "BT-TC-Network Operations",
        "BT-TC-Data Analytics"
    ]
    
    quarters = ["q4"]
    year = 2024
    
    for tech_center in tech_centers:
        for quarter in quarters:
            try:
                results = pipeline.run_training_with_domains(
                    tech_center=tech_center,
                    quarter=quarter,
                    year=year
                )
                
                print(f"✅ Training completed for {tech_center} Q{quarter}")
                
            except Exception as e:
                print(f"❌ Training failed for {tech_center} Q{quarter}: {e}")


if __name__ == "__main__":
    main()