"""
Training Pipeline Module
Orchestrates the complete training process: data loading, UMAP reduction, HDBSCAN clustering, and model storage
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import hashlib
import json
from pathlib import Path

from config.config import load_config
from data_access.bigquery_client import BigQueryClient
from training.umap import UMAPProcessor
from training.hdbscan_clustering import HDBSCANClusterer
from analysis.cluster_labeling import EnhancedClusterLabeler
from azure.storage.blob import BlobServiceClient

class TrainingPipeline:
    """
    Main training pipeline that orchestrates UMAP + HDBSCAN clustering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize training pipeline"""
        self.config = config if config is not None else load_config()
        self.bq_client = BigQueryClient(self.config)
        self.umap_processor = UMAPProcessor(self.config)
        self.hdbscan_clusterer = HDBSCANClusterer(self.config)
        self.cluster_labeler = EnhancedClusterLabeler(self.config)
        self.blob_client = self._initialize_blob_client()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_blob_client(self) -> Optional[BlobServiceClient]:
        """Initialize Azure Blob Storage client if enabled"""
        try:
            if not self.config.get('model_storage', {}).get('blob_storage_enabled', False):
                return None
                
            connection_string = self.config.get('azure', {}).get('blob_storage', {}).get('connection_string', '')
            if connection_string:
                return BlobServiceClient.from_connection_string(connection_string)
            return None
        except Exception as e:
            self.logger.warning(f"Could not initialize blob client: {e}")
            return None
        
    async def run_training(self, tech_center: str, training_date: datetime, 
                          months_back: int = 24) -> Dict[str, Any]:
        """Run complete training pipeline for a tech center"""
        try:
            self.logger.info(f"🚀 Starting training pipeline for {tech_center}")
            
            # Step 1: Load preprocessed data
            self.logger.info("📊 Loading preprocessed data from BigQuery...")
            incidents_df, embeddings = await self._load_training_data(
                tech_center, training_date, months_back
            )
            
            if incidents_df.empty:
                raise ValueError(f"No training data found for {tech_center}")
                
            self.logger.info(f"✅ Loaded {len(incidents_df)} incidents with embeddings")
            
            # Step 2: UMAP dimensionality reduction
            self.logger.info("🔄 Applying UMAP dimensionality reduction...")
            reduced_embeddings, umap_metrics = self.umap_processor.fit_transform(embeddings)
            
            # Step 3: HDBSCAN clustering
            self.logger.info("🔄 Applying HDBSCAN clustering...")
            cluster_labels, clustering_metrics = self.hdbscan_clusterer.fit_predict(reduced_embeddings)
            
            # Step 4: Validate results
            if not self.hdbscan_clusterer.validate_clustering_results(cluster_labels, clustering_metrics):
                raise ValueError("Clustering results failed quality validation")
            
            # Step 5: Generate cluster labels and domain grouping using LLM
            self.logger.info("🏷️ Generating cluster labels and domain grouping...")
            analysis_results = self.cluster_labeler.run_complete_analysis(
                incidents_df, cluster_labels, reduced_embeddings
            )
            
            # Step 6: Generate model version and hash
            model_version, model_hash = self._generate_model_version(
                tech_center, training_date, clustering_metrics
            )
            
            # Step 7: Save models and upload to blob storage
            self.logger.info("💾 Saving models and uploading to blob storage...")
            local_paths, blob_paths = await self._save_and_upload_models(
                model_version, model_hash, reduced_embeddings, tech_center
            )
            
            # Step 8: Store clustering results in BigQuery
            self.logger.info("💾 Storing clustering results in BigQuery...")
            await self._store_clustering_results(
                incidents_df, cluster_labels, reduced_embeddings, 
                model_version, model_hash, analysis_results['labeled_dataframe']
            )
            
            # Step 9: Register model in model registry
            self.logger.info("📋 Registering model in model registry...")
            await self._register_model(
                tech_center, model_version, model_hash, training_date,
                months_back, clustering_metrics, umap_metrics, blob_paths or local_paths
            )
            
            # Compile training results
            training_results = {
                'tech_center': tech_center,
                'model_version': model_version,
                'model_hash': model_hash,
                'training_date': training_date,
                'incidents_count': len(incidents_df),
                'clusters_found': clustering_metrics['n_clusters'],
                'domains_found': analysis_results['summary'].get('total_domains', 0),
                'noise_ratio': clustering_metrics['noise_ratio'],
                'silhouette_score': clustering_metrics.get('silhouette_score'),
                'local_paths': local_paths,
                'blob_paths': blob_paths,
                'umap_metrics': umap_metrics,
                'clustering_metrics': clustering_metrics,
                'cluster_analysis': analysis_results['summary'],
                'domain_names': analysis_results['summary'].get('domain_names', [])
            }
            
            self.logger.info("🎉 Training pipeline completed successfully!")
            self.logger.info(f"📊 Results: {clustering_metrics['n_clusters']} clusters grouped into {analysis_results['summary'].get('total_domains', 0)} domains")
            self.logger.info(f"🏷️ Domain names: {', '.join(analysis_results['summary'].get('domain_names', []))}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Training pipeline failed: {str(e)}")
            raise
    
    async def _load_training_data(self, tech_center: str, training_date: datetime, 
                                months_back: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load preprocessed incidents and embeddings from BigQuery using config query"""
        try:
            # Calculate start date
            start_date = training_date - timedelta(days=months_back * 30)
            
            # Get training data query from config
            query_template = self.config.get('bigquery', {}).get('queries', {}).get('training_data_window', '')
            if not query_template:
                raise ValueError("training_data_window query not found in configuration")
            
            # Get table references from config
            preprocessed_incidents_table = self.config.get('bigquery', {}).get('tables', {}).get('preprocessed_incidents', '')
            team_services_table = self.config.get('bigquery', {}).get('tables', {}).get('team_services', '')
            
            if not preprocessed_incidents_table:
                raise ValueError("preprocessed_incidents table not found in configuration")
            if not team_services_table:
                raise ValueError("team_services table not found in configuration")
            
            # Format the query with actual values
            query = query_template.format(
                preprocessed_incidents_table=preprocessed_incidents_table,
                team_services_table=team_services_table,
                start_date=start_date.strftime('%Y-%m-%d %H:%M:%S'),
                end_date=training_date.strftime('%Y-%m-%d %H:%M:%S'),
                tech_center=tech_center
            )
            
            self.logger.info(f"Executing training data query for date range: {start_date} to {training_date}")
            
            # Execute query
            incidents_df = await self.bq_client.execute_query(query)
            
            if incidents_df.empty:
                raise ValueError(f"No preprocessed incidents found for {tech_center} "
                               f"between {start_date} and {training_date}")
            
            # Validate required columns exist
            required_columns = ['embedding']
            missing_columns = [col for col in required_columns if col not in incidents_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in query result: {missing_columns}")
            
            # Extract embeddings and convert to numpy array
            embeddings_list = incidents_df['embedding'].tolist()
            embeddings = np.array(embeddings_list)
            
            # Validate embeddings
            if not self.umap_processor.validate_embeddings(embeddings):
                raise ValueError("Invalid embeddings in training data")
            
            self.logger.info(f"Loaded {len(incidents_df)} incidents with {embeddings.shape[1]}-d embeddings")
            
            return incidents_df, embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to load training data: {str(e)}")
            raise
    
    def _generate_model_version(self, tech_center: str, training_date: datetime, 
                              metrics: Dict[str, Any]) -> Tuple[str, str]:
        """Generate model version and hash"""
        # Generate version string
        year = training_date.year
        quarter = f"q{(training_date.month - 1) // 3 + 1}"
        model_version = f"{year}_{quarter}"
        
        # Generate hash based on model parameters and data characteristics
        hash_input = {
            "tech_center": tech_center,
            "training_date": training_date.isoformat()[:10],
            "incidents_count": metrics.get('total_points', 0),
            "umap_config": self.umap_processor.get_model_params(),
            "hdbscan_config": self.hdbscan_clusterer.get_model_params()
        }
        
        hash_string = json.dumps(hash_input, sort_keys=True)
        model_hash = hashlib.md5(hash_string.encode()).hexdigest()[:8]
        
        return model_version, model_hash
    
    async def _save_and_upload_models(self, model_version: str, model_hash: str, 
                                     reduced_embeddings: np.ndarray, tech_center: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Save models locally and upload to Azure Blob Storage"""
        try:
            # Save models locally first
            local_paths = await self._save_models_locally(model_version, model_hash, reduced_embeddings)
            
            # Upload to blob storage if enabled
            blob_paths = {}
            if self.blob_client:
                blob_paths = await self._upload_to_blob_storage(local_paths, model_version, model_hash, tech_center)
            
            return local_paths, blob_paths
            
        except Exception as e:
            self.logger.error(f"Failed to save and upload models: {str(e)}")
            raise
    
    async def _save_models_locally(self, model_version: str, model_hash: str, 
                                 reduced_embeddings: np.ndarray) -> Dict[str, str]:
        """Save trained models to local storage (no embeddings - they're in BigQuery)"""
        try:
            models_base_dir = self.config.get('model_storage', {}).get('local_path', 'models')
            models_dir = Path(models_base_dir) / f"{model_version}_{model_hash}"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            local_paths = {
                'umap_model': str(models_dir / f"umap_model_{model_hash}.pkl"),
                'hdbscan_model': str(models_dir / f"hdbscan_model_{model_hash}.pkl"),
                'metadata': str(models_dir / f"model_metadata_{model_hash}.json")
            }
            
            # Save UMAP model
            if not self.umap_processor.save_model(local_paths['umap_model']):
                raise ValueError("Failed to save UMAP model")
            
            # Save HDBSCAN model
            if not self.hdbscan_clusterer.save_model(local_paths['hdbscan_model']):
                raise ValueError("Failed to save HDBSCAN model")
            
            # Save model metadata
            metadata = {
                'model_version': model_version,
                'model_hash': model_hash,
                'created_timestamp': datetime.now().isoformat(),
                'umap_params': self.umap_processor.get_model_params(),
                'hdbscan_params': self.hdbscan_clusterer.get_model_params(),
                'training_data_shape': reduced_embeddings.shape,
                'embeddings_source': 'bigquery_preprocessed_incidents',
                'file_paths': local_paths
            }
            
            with open(local_paths['metadata'], 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"✅ Models saved locally to: {models_dir}")
            return local_paths
            
        except Exception as e:
            self.logger.error(f"Failed to save models locally: {str(e)}")
            raise
    
    async def _upload_to_blob_storage(self, local_paths: Dict[str, str], 
                                    model_version: str, model_hash: str, tech_center: str) -> Dict[str, str]:
        """Upload model artifacts to Azure Blob Storage"""
        try:
            container_name = self.config.get('azure', {}).get('blob_storage', {}).get('container_name', 'hdbscan-models')
            
            # Create tech center slug for blob path
            tech_center_slug = tech_center.lower().replace(' ', '-').replace(',', '').replace('&', 'and')
            blob_base_path = f"{tech_center_slug}/{model_version}_{model_hash}"
            
            blob_paths = {}
            
            for artifact_type, local_path in local_paths.items():
                filename = Path(local_path).name
                blob_name = f"{blob_base_path}/{filename}"
                
                blob_client = self.blob_client.get_blob_client(
                    container=container_name, 
                    blob=blob_name
                )
                
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                
                blob_paths[artifact_type] = f"https://{self.blob_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
                
                self.logger.info(f"✅ Uploaded {artifact_type}: {blob_name}")
            
            return blob_paths
            
        except Exception as e:
            self.logger.error(f"Failed to upload to blob storage: {str(e)}")
            raise
    
    async def _store_clustering_results(self, incidents_df: pd.DataFrame, 
                                      cluster_labels: np.ndarray,
                                      reduced_embeddings: np.ndarray,
                                      model_version: str, model_hash: str,
                                      labeled_df: pd.DataFrame):
        """Store clustering results in BigQuery"""
        try:
            # Create enhanced results dataframe
            results_df = labeled_df.copy()
            results_df['umap_x'] = reduced_embeddings[:, 0]
            results_df['umap_y'] = reduced_embeddings[:, 1]
            results_df['model_version'] = model_version
            results_df['model_hash'] = model_hash
            results_df['prediction_timestamp'] = datetime.now()
            
            # Create versioned table name
            cluster_table_name = f"cluster_results_{model_version}_{model_hash}"
            project_id = self.config.get('bigquery', {}).get('project_id', '')
            dataset_id = self.config.get('bigquery', {}).get('dataset_id', '')
            cluster_table_id = f"{project_id}.{dataset_id}.{cluster_table_name}"
            
            # Store results
            success = await self.bq_client.store_dataframe(cluster_table_id, results_df)
            
            if not success:
                raise ValueError("Failed to store clustering results in BigQuery")
                
            self.logger.info(f"Stored clustering results in table: {cluster_table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store clustering results: {str(e)}")
            raise
    
    async def _register_model(self, tech_center: str, model_version: str, 
                            model_hash: str, training_date: datetime,
                            months_back: int, clustering_metrics: Dict[str, Any],
                            umap_metrics: Dict[str, Any], artifact_paths: Dict[str, str]):
        """Register model in the model registry"""
        try:
            start_date = training_date - timedelta(days=months_back * 30)
            
            model_metadata = {
                'model_version': model_version,
                'tech_center': tech_center,
                'model_type': 'hdbscan_umap',
                'training_data_start': start_date.date(),
                'training_data_end': training_date.date(),
                'blob_path': json.dumps(artifact_paths),
                'created_timestamp': datetime.now(),
                'model_params': json.dumps({
                    'umap': self.umap_processor.get_model_params(),
                    'hdbscan': self.hdbscan_clusterer.get_model_params()
                }),
                'cluster_count': clustering_metrics['n_clusters'],
                'silhouette_score': clustering_metrics.get('silhouette_score')
            }
            
            success = await self.bq_client.register_model_version(model_metadata)
            
            if not success:
                raise ValueError("Failed to register model in model registry")
                
            self.logger.info(f"Registered model {model_version}_{model_hash} in model registry")
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {str(e)}")
            raise


# Example usage
async def main():
    """Example of how to run the training pipeline"""
    
    # Initialize pipeline
    pipeline = TrainingPipeline()
    
    # Run training for a specific tech center
    tech_center = "BT-TC-AI, Analytics & Data"
    training_date = datetime(2024, 12, 31)
    months_back = 24
    
    try:
        results = await pipeline.run_training(tech_center, training_date, months_back)
        print(f"✅ Training completed successfully!")
        print(f"📊 Model: {results['model_version']}_{results['model_hash']}")
        print(f"🎯 Clusters: {results['clusters_found']}")
        print(f"🗂️ Domains: {results['domains_found']}")
        print(f"📈 Silhouette Score: {results['silhouette_score']}")
        print(f"🏷️ Domains: {', '.join(results['domain_names'])}")
        
        if results.get('blob_paths'):
            print(f"☁️ Models uploaded to blob storage: {len(results['blob_paths'])} files")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())