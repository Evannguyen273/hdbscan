"""
Enhanced Cluster Analysis Module for HDBSCAN Pipeline
Supports versioned models and blob storage integration
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import Counter
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedClusterAnalyzer:
    """
    Enhanced cluster analysis for versioned HDBSCAN models with blob storage integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cluster analyzer with configuration
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config
        self.analysis_results = {}
        
    def analyze_clustering_results(
        self, 
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        umap_embeddings: np.ndarray,
        tech_center: str,
        model_version: str
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of clustering results for versioned models
        
        Args:
            embeddings: Original high-dimensional embeddings
            cluster_labels: HDBSCAN cluster assignments
            umap_embeddings: UMAP reduced embeddings
            tech_center: Technology center name
            model_version: Model version (e.g., "2025_q2")
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting cluster analysis for {tech_center} - {model_version}")
        
        analysis = {
            "tech_center": tech_center,
            "model_version": model_version,
            "analysis_timestamp": datetime.now().isoformat(),
            "cluster_statistics": self._calculate_cluster_statistics(cluster_labels),
            "quality_metrics": self._calculate_quality_metrics(embeddings, umap_embeddings, cluster_labels),
            "cluster_characteristics": self._analyze_cluster_characteristics(cluster_labels),
            "outlier_analysis": self._analyze_outliers(cluster_labels),
            "stability_metrics": self._calculate_stability_metrics(cluster_labels),
        }
        
        # Store results for potential comparison
        self.analysis_results[f"{tech_center}_{model_version}"] = analysis
        
        logger.info(f"Cluster analysis completed for {tech_center} - {model_version}")
        return analysis
    
    def _calculate_cluster_statistics(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate basic cluster statistics"""
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise (-1)
        n_noise = np.sum(cluster_labels == -1)
        n_points = len(cluster_labels)
        
        cluster_sizes = Counter(cluster_labels[cluster_labels != -1])
        
        stats = {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "n_total_points": n_points,
            "noise_ratio": n_noise / n_points if n_points > 0 else 0,
            "avg_cluster_size": np.mean(list(cluster_sizes.values())) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
            "cluster_size_distribution": dict(cluster_sizes)
        }
        
        logger.debug(f"Cluster statistics: {n_clusters} clusters, {n_noise} noise points")
        return stats
    
    def _calculate_quality_metrics(
        self, 
        embeddings: np.ndarray, 
        umap_embeddings: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics"""
        metrics = {}
        
        # Filter out noise points for quality metrics
        mask = cluster_labels != -1
        if np.sum(mask) < 2:
            logger.warning("Not enough non-noise points for quality metrics")
            return {"silhouette_score": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": float('inf')}
        
        filtered_embeddings = embeddings[mask]
        filtered_umap = umap_embeddings[mask]
        filtered_labels = cluster_labels[mask]
        
        # Skip if only one cluster
        if len(np.unique(filtered_labels)) < 2:
            logger.warning("Less than 2 clusters found for quality metrics")
            return {"silhouette_score": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": float('inf')}
        
        try:
            # Silhouette score (higher is better, range: -1 to 1)
            metrics["silhouette_score"] = silhouette_score(filtered_umap, filtered_labels)
            
            # Calinski-Harabasz score (higher is better)
            metrics["calinski_harabasz"] = calinski_harabasz_score(filtered_umap, filtered_labels)
            
            # Davies-Bouldin score (lower is better)
            metrics["davies_bouldin"] = davies_bouldin_score(filtered_umap, filtered_labels)
            
            logger.debug(f"Quality metrics calculated: silhouette={metrics['silhouette_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            metrics = {"silhouette_score": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": float('inf')}
        
        return metrics
    
    def _analyze_cluster_characteristics(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of each cluster"""
        unique_labels = np.unique(cluster_labels)
        characteristics = {}
        
        for label in unique_labels:
            if label == -1:  # Noise cluster
                characteristics["noise"] = {
                    "size": np.sum(cluster_labels == -1),
                    "percentage": (np.sum(cluster_labels == -1) / len(cluster_labels)) * 100
                }
            else:
                cluster_mask = cluster_labels == label
                cluster_size = np.sum(cluster_mask)
                
                characteristics[f"cluster_{label}"] = {
                    "size": cluster_size,
                    "percentage": (cluster_size / len(cluster_labels)) * 100,
                    "density_rank": None,  # Could be calculated if needed
                }
        
        return characteristics
    
    def _analyze_outliers(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze outlier detection results"""
        n_outliers = np.sum(cluster_labels == -1)
        n_total = len(cluster_labels)
        
        outlier_analysis = {
            "n_outliers": n_outliers,
            "outlier_percentage": (n_outliers / n_total) * 100 if n_total > 0 else 0,
            "outlier_threshold": "HDBSCAN_auto",  # HDBSCAN handles this automatically
        }
        
        # Outlier percentage quality assessment
        outlier_pct = outlier_analysis["outlier_percentage"]
        if outlier_pct < 5:
            quality = "excellent"
        elif outlier_pct < 10:
            quality = "good"
        elif outlier_pct < 20:
            quality = "acceptable"
        else:
            quality = "poor"
            
        outlier_analysis["quality_assessment"] = quality
        
        return outlier_analysis
    
    def _calculate_stability_metrics(self, cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate cluster stability metrics"""
        unique_labels = np.unique(cluster_labels[cluster_labels != -1])
        
        if len(unique_labels) == 0:
            return {"stability_score": 0.0, "cluster_balance": 0.0}
        
        # Cluster size balance (entropy-based)
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        total_points = sum(cluster_sizes)
        
        if total_points == 0:
            return {"stability_score": 0.0, "cluster_balance": 0.0}
        
        proportions = [size / total_points for size in cluster_sizes]
        entropy = -sum(p * np.log(p) for p in proportions if p > 0)
        max_entropy = np.log(len(proportions)) if len(proportions) > 1 else 1
        
        balance = entropy / max_entropy if max_entropy > 0 else 0
        
        # Simple stability score based on cluster count and balance
        stability = min(1.0, len(unique_labels) / 20) * balance  # Normalize by expected max clusters
        
        return {
            "stability_score": stability,
            "cluster_balance": balance,
            "entropy": entropy
        }
    
    def compare_model_versions(
        self, 
        tech_center: str, 
        version1: str, 
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare analysis results between two model versions
        
        Args:
            tech_center: Technology center name
            version1: First model version
            version2: Second model version
            
        Returns:
            Comparison results
        """
        key1 = f"{tech_center}_{version1}"
        key2 = f"{tech_center}_{version2}"
        
        if key1 not in self.analysis_results or key2 not in self.analysis_results:
            logger.error(f"Analysis results not found for comparison: {key1}, {key2}")
            return {}
        
        analysis1 = self.analysis_results[key1]
        analysis2 = self.analysis_results[key2]
        
        comparison = {
            "tech_center": tech_center,
            "version1": version1,
            "version2": version2,
            "cluster_count_change": (
                analysis2["cluster_statistics"]["n_clusters"] - 
                analysis1["cluster_statistics"]["n_clusters"]
            ),
            "quality_score_change": (
                analysis2["quality_metrics"]["silhouette_score"] - 
                analysis1["quality_metrics"]["silhouette_score"]
            ),
            "noise_ratio_change": (
                analysis2["cluster_statistics"]["noise_ratio"] - 
                analysis1["cluster_statistics"]["noise_ratio"]
            ),
            "recommendation": self._generate_comparison_recommendation(analysis1, analysis2)
        }
        
        logger.info(f"Model comparison completed for {tech_center}: {version1} vs {version2}")
        return comparison
    
    def _generate_comparison_recommendation(
        self, 
        analysis1: Dict[str, Any], 
        analysis2: Dict[str, Any]
    ) -> str:
        """Generate recommendation based on comparison"""
        score1 = analysis1["quality_metrics"]["silhouette_score"]
        score2 = analysis2["quality_metrics"]["silhouette_score"]
        
        noise1 = analysis1["cluster_statistics"]["noise_ratio"]
        noise2 = analysis2["cluster_statistics"]["noise_ratio"]
        
        if score2 > score1 + 0.05 and noise2 < noise1:
            return "upgrade_recommended"
        elif score2 < score1 - 0.05 or noise2 > noise1 + 0.1:
            return "keep_previous_version"
        else:
            return "similar_performance"
    
    def generate_analysis_report(self, tech_center: str, model_version: str) -> str:
        """
        Generate a comprehensive analysis report
        
        Args:
            tech_center: Technology center name
            model_version: Model version
            
        Returns:
            Formatted analysis report
        """
        key = f"{tech_center}_{model_version}"
        if key not in self.analysis_results:
            return f"No analysis results found for {tech_center} - {model_version}"
        
        analysis = self.analysis_results[key]
        
        report = f"""
CLUSTER ANALYSIS REPORT
=======================
Tech Center: {tech_center}
Model Version: {model_version}
Analysis Date: {analysis['analysis_timestamp']}

CLUSTER STATISTICS:
- Number of Clusters: {analysis['cluster_statistics']['n_clusters']}
- Noise Points: {analysis['cluster_statistics']['n_noise_points']} ({analysis['cluster_statistics']['noise_ratio']:.1%})
- Average Cluster Size: {analysis['cluster_statistics']['avg_cluster_size']:.1f}

QUALITY METRICS:
- Silhouette Score: {analysis['quality_metrics']['silhouette_score']:.3f}
- Calinski-Harabasz Score: {analysis['quality_metrics']['calinski_harabasz']:.1f}
- Davies-Bouldin Score: {analysis['quality_metrics']['davies_bouldin']:.3f}

OUTLIER ANALYSIS:
- Outlier Percentage: {analysis['outlier_analysis']['outlier_percentage']:.1f}%
- Quality Assessment: {analysis['outlier_analysis']['quality_assessment']}

STABILITY METRICS:
- Stability Score: {analysis['stability_metrics']['stability_score']:.3f}
- Cluster Balance: {analysis['stability_metrics']['cluster_balance']:.3f}
"""
        
        return report
    
    def save_analysis_results(self, filepath: str) -> None:
        """Save analysis results to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            logger.info(f"Analysis results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    def load_analysis_results(self, filepath: str) -> None:
        """Load analysis results from JSON file"""
        try:
            with open(filepath, 'r') as f:
                self.analysis_results = json.load(f)
            logger.info(f"Analysis results loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading analysis results: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Test the cluster analyzer
    from logging_setup import setup_detailed_logging
    setup_detailed_logging()
    
    # Mock data for testing
    embeddings = np.random.rand(1000, 1536)
    umap_embeddings = np.random.rand(1000, 2)
    cluster_labels = np.random.choice([-1, 0, 1, 2, 3, 4], 1000)
    
    config = {"analysis": {"min_cluster_size": 10}}
    analyzer = EnhancedClusterAnalyzer(config)
    
    results = analyzer.analyze_clustering_results(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        umap_embeddings=umap_embeddings,
        tech_center="BT-TC-Data Analytics",
        model_version="2025_q2"
    )
    
    print(analyzer.generate_analysis_report("BT-TC-Data Analytics", "2025_q2"))