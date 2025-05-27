import logging
import json
import pandas as pd
import os
from typing import Dict, List, Optional

class ClusterAnalyzer:
    def __init__(self, config):
        self.config = config
    
    def generate_cluster_info(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'short_description', 
        cluster_column: str = 'cluster', 
        sample_size: int = 5, 
        output_dir: Optional[str] = None
    ) -> Dict:
        """Generate detailed information about each cluster"""
        
        logging.info(f"Generating cluster information from {len(df)} records")
        
        # Count instances of each cluster
        cluster_counts = df[cluster_column].value_counts().to_dict()
        total_records = len(df)
        
        # Initialize results dictionary
        clusters_info = {}
        
        # Process each cluster
        for cluster_id, count in cluster_counts.items():
            # Calculate percentage
            percentage = round((count / total_records) * 100, 2)
            
            # Get samples
            cluster_mask = df[cluster_column] == cluster_id
            samples = df[cluster_mask].sample(min(sample_size, count))[text_column].tolist()
            
            # Store information
            cluster_key = str(cluster_id)  # Convert to string for JSON compatibility
            clusters_info[cluster_key] = {
                "size": int(count),
                "percentage": percentage,
                "samples": samples
            }
        
        # Save to file if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/cluster_details.json", "w") as f:
                json.dump(clusters_info, f, indent=2)
            logging.info(f"Saved cluster details to {output_dir}/cluster_details.json")
        
        return clusters_info
    
    def apply_labels_to_data(self, df: pd.DataFrame, labeled_clusters: Dict, domains: Dict) -> pd.DataFrame:
        """Apply cluster labels and domains to the data"""
        
        result_df = df.copy()
        
        # Create mapping dictionaries for efficient lookups
        topic_mapping = {self._safe_cluster_id_to_int(k): v["topic"] for k, v in labeled_clusters.items() if k != "-1"}
        topic_mapping[-1] = "Noise"  # Add mapping for noise cluster
        
        # Create domain mapping
        domain_mapping = {}
        for domain in domains.get("domains", []):
            domain_name = domain["domain_name"]
            for cluster_id in domain.get("clusters", []):
                domain_mapping[int(cluster_id)] = domain_name
        
        # Apply mappings to create category and subcategory columns
        result_df["subcategory"] = result_df["cluster"].map(topic_mapping).fillna("Unknown")
        result_df["category"] = result_df["cluster"].map(domain_mapping).fillna("Other")
        
        return result_df
    
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