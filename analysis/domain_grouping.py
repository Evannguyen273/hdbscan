import logging
import json
import time
import uuid
import numpy as np
import openai
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, Optional
import os

class DomainGrouper:
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
    
    def _safe_cluster_id_to_int(self, cid):
        """Helper function to safely convert cluster IDs to integers"""
        try:
            return int(cid)
        except ValueError:
            if isinstance(cid, str) and 'cluster_' in cid:
                try:
                    return int(cid.split('_')[-1])
                except (ValueError, IndexError):
                    logging.warning(f"Could not convert cluster ID '{cid}' to integer. Using -99 as placeholder.")
                    return -99
            else:
                logging.warning(f"Could not convert cluster ID '{cid}' to integer. Using -99 as placeholder.")
                return -99
    
    def group_clusters_into_domains(
        self,
        labeled_clusters: Dict,
        clusters_info: Dict,
        umap_embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        output_dir: Optional[str] = None
    ) -> Dict:
        """Group clusters into domains using coordinate proximity and semantic similarity"""
        
        max_domains = self.config.clustering.max_domains
        logging.info(f"Grouping clusters into domains with auto-optimization (max: {max_domains} domains)")
        
        # Skip noise cluster for domain grouping
        clusters_to_group = {k: v for k, v in labeled_clusters.items() if k != "-1"}
        
        # If very few clusters, don't bother with domains
        if len(clusters_to_group) <= 3:
            domains = {
                "domains": [
                    {
                        "domain_name": "All Incidents",
                        "description": "All incident clusters",
                        "clusters": [self._safe_cluster_id_to_int(k) for k in clusters_to_group.keys()]
                    }
                ]
            }
            domains["domains"].append({"domain_name": "Noise", "description": "Uncategorized incidents", "clusters": [-1]})
            
            if output_dir:
                with open(f"{output_dir}/domains.json", "w") as f:
                    json.dump(domains, f, indent=2)
            
            return domains
        
        # Calculate cluster centroids in UMAP space
        unique_clusters = np.unique(cluster_labels)
        cluster_centroids = {}
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise
                continue
            cluster_mask = cluster_labels == cluster_id
            cluster_points = umap_embeddings[cluster_mask]
            if len(cluster_points) > 0:
                cluster_centroids[cluster_id] = np.mean(cluster_points, axis=0)
        
        # Convert centroids to array for hierarchical clustering
        centroid_ids = list(cluster_centroids.keys())
        centroid_coords = np.array([cluster_centroids[cid] for cid in centroid_ids])
        
        # Determine optimal number of domains
        max_domains_to_try = min(max_domains, len(centroid_ids) - 1)
        max_domains_to_try = max(2, max_domains_to_try)
        
        scores = []
        domain_counts = range(2, max_domains_to_try + 1)
        
        logging.info("Determining optimal number of domains...")
        for n_domains in domain_counts:
            hc = AgglomerativeClustering(n_clusters=n_domains, linkage='ward')
            domain_labels = hc.fit_predict(centroid_coords)
            
            if len(np.unique(domain_labels)) > 1:
                try:
                    sil_score = silhouette_score(centroid_coords, domain_labels)
                    ch_score = calinski_harabasz_score(centroid_coords, domain_labels)
                    combined_score = sil_score + (ch_score / (1000 + ch_score))
                    scores.append(combined_score)
                    logging.info(f"  {n_domains} domains: silhouette={sil_score:.3f}, CH={ch_score:.1f}, combined={combined_score:.3f}")
                except Exception as e:
                    logging.warning(f"Error calculating score for {n_domains} domains: {e}")
                    scores.append(-1)
            else:
                scores.append(-1)
        
        # Find optimal number of domains
        valid_scores = [(i, score) for i, score in enumerate(scores) if score > 0]
        if valid_scores:
            best_idx, best_score = max(valid_scores, key=lambda x: x[1])
            optimal_domains = domain_counts[best_idx]
            logging.info(f"Selected optimal domain count: {optimal_domains} (score: {best_score:.3f})")
        else:
            optimal_domains = min(15, max(2, len(centroid_ids) // 5))
            logging.warning(f"Could not determine optimal domains. Using default: {optimal_domains}")
        
        # Apply hierarchical clustering with optimal domain count
        hc = AgglomerativeClustering(n_clusters=optimal_domains, linkage='ward')
        domain_labels = hc.fit_predict(centroid_coords)
        
        # Create initial domain mapping
        initial_domains = {}
        for i, domain_id in enumerate(domain_labels):
            if domain_id not in initial_domains:
                initial_domains[domain_id] = []
            initial_domains[domain_id].append(centroid_ids[i])
        
        # Remove empty domains
        initial_domains = {k: v for k, v in initial_domains.items() if v}
        domain_count = len(initial_domains)
        
        logging.info(f"Initial domain grouping created {domain_count} domains based on cluster coordinates")
        
        # Use LLM to name and refine domains
        final_domains = []
        all_standardized_labels = {}
        
        for domain_id, cluster_ids in initial_domains.items():
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
    "{cluster_ids[1] if len(cluster_ids) > 1 else cluster_ids[0]}": "Standardized topic label for cluster {cluster_ids[1] if len(cluster_ids) > 1 else cluster_ids[0]}",
    ... for all clusters in the domain
  }}
}}

IMPORTANT: For all cluster IDs in standardized_labels, use ONLY NUMERIC values without any prefixes.
DO NOT use formats like "cluster_167" - use ONLY "167" as the key.

Ensure standardized labels maintain specific system names where present but use consistent terminology.
"""
                
                # Generate correlation ID for Azure tracking
                correlation_id = str(uuid.uuid4())
                
                # Call Azure OpenAI with retry logic
                retry_attempts = 3
                domain_result = None
                
                for attempt in range(retry_attempts):
                    try:
                        logging.info(f"Processing domain {domain_id+1}/{domain_count} with {len(cluster_ids)} clusters (attempt {attempt+1})")
                        
                        response = self.openai_client.chat.completions.create(
                            model=self.config.azure.chat_model,
                            messages=[
                                {"role": "system", "content": "You are an IT domain classification expert. Return properly structured JSON with no additional text. Use numeric IDs only without any prefixes like 'cluster_'."},
                                {"role": "user", "content": prompt_text}
                            ],
                            temperature=0.3,
                            response_format={"type": "json_object"},
                            timeout=30,
                            user=correlation_id
                        )
                        
                        # Parse response
                        result_text = response.choices[0].message.content
                        domain_result = json.loads(result_text)
                        
                        # Validate expected structure
                        if "domain_name" in domain_result and "description" in domain_result and "standardized_labels" in domain_result:
                            break
                        else:
                            missing = []
                            if "domain_name" not in domain_result:
                                missing.append("domain_name")
                            if "description" not in domain_result:
                                missing.append("description")
                            if "standardized_labels" not in domain_result:
                                missing.append("standardized_labels")
                            raise ValueError(f"Response missing required keys: {missing}")
                        
                    except Exception as e:
                        if attempt < retry_attempts - 1:
                            wait_time = min(30, 2 ** attempt * 2)
                            logging.warning(f"Domain processing attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                            time.sleep(wait_time)
                        else:
                            # Use fallback for this domain if all attempts fail
                            logging.error(f"Domain processing failed after all attempts: {e}")
                            if cluster_ids:
                                sample_cluster = str(cluster_ids[0])
                                sample_topic = labeled_clusters.get(sample_cluster, {}).get("topic", f"Cluster {cluster_ids[0]}")
                                domain_result = {
                                    "domain_name": f"Domain {domain_id+1}",
                                    "description": f"Related incidents including {sample_topic}",
                                    "standardized_labels": {}
                                }
                            else:
                                domain_result = {
                                    "domain_name": f"Domain {domain_id+1}",
                                    "description": "Group of related incidents",
                                    "standardized_labels": {}
                                }
                
                # Add the domain to our results
                if domain_result and cluster_ids:
                    final_domains.append({
                        "domain_name": domain_result["domain_name"],
                        "description": domain_result["description"],
                        "clusters": [self._safe_cluster_id_to_int(cid) for cid in cluster_ids]
                    })
                    
                    # Add standardized labels to the overall collection
                    if "standardized_labels" in domain_result:
                        for cluster_id, new_label in domain_result["standardized_labels"].items():
                            clean_cluster_id = str(self._safe_cluster_id_to_int(cluster_id))
                            all_standardized_labels[clean_cluster_id] = new_label
                            
            except Exception as e:
                logging.error(f"Error processing domain {domain_id}: {e}")
                if not cluster_ids:
                    continue
                
                # Add a fallback domain
                final_domains.append({
                    "domain_name": f"Domain {domain_id}",
                    "description": "Group of related incidents",
                    "clusters": [int(cid) for cid in cluster_ids]
                })
        
        # Add the noise domain
        final_domains.append({
            "domain_name": "Noise",
            "description": "Uncategorized incidents",
            "clusters": [-1]
        })
        
        # Create the final result structure
        domains = {"domains": final_domains}
        
        # Apply standardized labels to the labeled_clusters dictionary
        for cluster_id, new_label in all_standardized_labels.items():
            if cluster_id in labeled_clusters:
                labeled_clusters[cluster_id]["topic"] = new_label
        
        # Save results
        if output_dir:
            with open(f"{output_dir}/domains.json", "w") as f:
                json.dump(domains, f, indent=2)
            
            if all_standardized_labels:
                with open(f"{output_dir}/labeled_clusters_standardized.json", "w") as f:
                    json.dump(labeled_clusters, f, indent=2)
            
            # Save domain metrics
            metrics = {
                "optimal_domain_count": optimal_domains,
                "tested_domain_counts": list(domain_counts),
                "domain_scores": [float(s) if s > 0 else None for s in scores],
                "final_domain_count": len(final_domains) - 1  # Exclude noise domain
            }
            with open(f"{output_dir}/domain_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        
        return domains