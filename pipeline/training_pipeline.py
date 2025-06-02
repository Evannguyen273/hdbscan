import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import json
import asyncio

from config.config import get_config
from clustering.hdbscan_clusterer import HDBSCANClusterer
from clustering.domain_grouper import DomainGrouper

class TrainingPipeline:
    """
    Training pipeline for cumulative HDBSCAN approach with versioned storage.
    Handles domain grouping and model versioning.
    """
    
    def __init__(self, config=None):
        """Initialize training pipeline with updated config system"""
        self.config = config if config is not None else get_config()
        
        # Initialize clustering components
        self.clusterer = HDBSCANClusterer(self.config)
        self.domain_grouper = DomainGrouper(self.config)
        
        # Training state
        self.training_stats = {}
        
        logging.info("Training pipeline initialized for cumulative approach")
    
    async def train_tech_center(self, tech_center: str, preprocessing_data: Dict, 
                               version: str, year: int, quarter: str) -> Dict:
        """
        Train HDBSCAN model for a specific tech center using cumulative approach.
        
        Args:
            tech_center: Tech center name
            preprocessing_data: Preprocessed embeddings and text data
            version: Model version (e.g., '2025_q2')
            year: Training year
            quarter: Training quarter
            
        Returns:
            Training results with model data and metrics
        """
        training_start = datetime.now()
        
        logging.info("Starting training for tech center: %s (version: %s)", tech_center, version)
        
        try:
            # Extract data from preprocessing results
            embeddings = preprocessing_data['embeddings']
            incident_data = preprocessing_data['incident_data']
            
            if len(embeddings) == 0:
                logging.warning("No embeddings available for %s", tech_center)
                return {"status": "failed", "reason": "no_embeddings"}
            
            # Stage 1: Domain grouping (if enabled)
            if self.config.clustering.domain_grouping['enabled']:
                logging.info("Stage 1: Performing domain grouping for %s", tech_center)
                domain_results = await self._perform_domain_grouping(
                    tech_center, incident_data, embeddings
                )
                
                if domain_results['status'] != 'success':
                    return domain_results
                
                training_data = domain_results['training_data']
            else:
                logging.info("Domain grouping disabled, using all data for %s", tech_center)
                training_data = [(incident_data, embeddings)]
            
            # Stage 2: Train HDBSCAN models
            logging.info("Stage 2: Training HDBSCAN models for %s", tech_center)
            model_results = await self._train_hdbscan_models(
                tech_center, training_data, version
            )
            
            # Stage 3: Generate model metadata and hash
            logging.info("Stage 3: Generating model metadata for %s", tech_center)
            model_metadata = self._generate_model_metadata(
                tech_center, model_results, version, year, quarter,
                len(incident_data), training_start
            )
            
            # Stage 4: Store training results in versioned BigQuery table
            logging.info("Stage 4: Storing training results for %s", tech_center)
            storage_results = await self._store_training_results(
                tech_center, model_results, model_metadata, version
            )
            
            training_duration = datetime.now() - training_start
            
            final_results = {
                "status": "success",
                "tech_center": tech_center,
                "version": version,
                "model_data": {
                    "models": model_results['models'],
                    "metadata": model_metadata,
                    "domain_info": model_results.get('domain_info', {})
                },
                "metrics": model_results['metrics'],
                "training_duration_seconds": training_duration.total_seconds(),
                "incidents_trained": len(incident_data),
                "storage_results": storage_results
            }
            
            logging.info("Training completed for %s in %.2f minutes", 
                        tech_center, training_duration.total_seconds() / 60)
            
            return final_results
            
        except Exception as e:
            logging.error("Training failed for %s: %s", tech_center, str(e))
            return {
                "status": "failed",
                "tech_center": tech_center,
                "version": version,
                "error": str(e),
                "training_duration_seconds": (datetime.now() - training_start).total_seconds()
            }
    
    async def _perform_domain_grouping(self, tech_center: str, 
                                     incident_data: pd.DataFrame, 
                                     embeddings: np.ndarray) -> Dict:
        """Perform domain grouping for tech center"""
        try:
            max_domains = self.config.clustering.domain_grouping['max_domains_per_tech_center']
            min_incidents = self.config.clustering.domain_grouping['min_incidents_per_domain']
            
            logging.info("Domain grouping: max_domains=%d, min_incidents=%d", 
                        max_domains, min_incidents)
            
            # Run domain grouping
            domain_results = self.domain_grouper.group_by_domains(
                incident_data, embeddings, 
                max_domains=max_domains,
                min_incidents_per_domain=min_incidents
            )
            
            if len(domain_results) == 0:
                logging.warning("Domain grouping produced no valid domains for %s", tech_center)
                return {"status": "failed", "reason": "no_valid_domains"}
            
            # Prepare training data by domain
            training_data = []
            for domain_name, domain_data in domain_results.items():
                domain_incidents = domain_data['incidents']
                domain_embeddings = domain_data['embeddings']
                
                logging.info("Domain '%s': %d incidents", domain_name, len(domain_incidents))
                training_data.append((domain_incidents, domain_embeddings, domain_name))
            
            return {
                "status": "success",
                "training_data": training_data,
                "domain_count": len(domain_results),
                "total_incidents": sum(len(data[0]) for data in training_data)
            }
            
        except Exception as e:
            logging.error("Domain grouping failed for %s: %s", tech_center, str(e))
            return {"status": "failed", "reason": "domain_grouping_error", "error": str(e)}
    
    async def _train_hdbscan_models(self, tech_center: str, 
                                  training_data: List[Tuple], 
                                  version: str) -> Dict:
        """Train HDBSCAN models for each domain/dataset"""
        models = {}
        metrics = {}
        domain_info = {}
        
        for i, data in enumerate(training_data):
            if len(data) == 3:  # Domain-grouped data
                incidents, embeddings, domain_name = data
                model_key = f"{tech_center}_{domain_name}"
            else:  # All data together
                incidents, embeddings = data
                domain_name = "all_incidents"
                model_key = f"{tech_center}_all"
            
            logging.info("Training HDBSCAN for %s (domain: %s, incidents: %d)", 
                        tech_center, domain_name, len(incidents))
            
            try:
                # Train HDBSCAN model
                model_result = await self.clusterer.fit_predict_async(
                    embeddings, incidents
                )
                
                # Store model and metrics
                models[model_key] = {
                    "model": model_result['model'],
                    "cluster_labels": model_result['labels'],
                    "probabilities": model_result.get('probabilities', []),
                    "domain_name": domain_name,
                    "incident_count": len(incidents)
                }
                
                metrics[model_key] = model_result.get('metrics', {})
                domain_info[domain_name] = {
                    "incident_count": len(incidents),
                    "cluster_count": len(set(model_result['labels'])) - (1 if -1 in model_result['labels'] else 0),
                    "noise_incidents": sum(1 for label in model_result['labels'] if label == -1)
                }
                
                logging.info("HDBSCAN trained for %s: %d clusters, %d noise", 
                           model_key, domain_info[domain_name]['cluster_count'], 
                           domain_info[domain_name]['noise_incidents'])
                
            except Exception as e:
                logging.error("HDBSCAN training failed for %s: %s", model_key, str(e))
                models[model_key] = {"status": "failed", "error": str(e)}
                metrics[model_key] = {"status": "failed"}
        
        return {
            "models": models,
            "metrics": metrics,
            "domain_info": domain_info,
            "total_models": len(models)
        }
    
    def _generate_model_metadata(self, tech_center: str, model_results: Dict, 
                               version: str, year: int, quarter: str,
                               total_incidents: int, training_start: datetime) -> Dict:
        """Generate comprehensive model metadata"""
        training_duration = datetime.now() - training_start
        
        # Calculate aggregate metrics
        successful_models = sum(1 for model in model_results['models'].values() 
                               if model.get('status') != 'failed')
        
        total_clusters = sum(info.get('cluster_count', 0) 
                            for info in model_results['domain_info'].values())
        
        total_noise = sum(info.get('noise_incidents', 0) 
                         for info in model_results['domain_info'].values())
        
        metadata = {
            "version": version,
            "tech_center": tech_center,
            "training_info": {
                "year": year,
                "quarter": quarter,
                "training_date": datetime.now().isoformat(),
                "training_duration_seconds": training_duration.total_seconds(),
                "training_window_months": self.config.training.training_window_months
            },
            "data_info": {
                "total_incidents": total_incidents,
                "incidents_processed": sum(model.get('incident_count', 0) 
                                         for model in model_results['models'].values()
                                         if model.get('status') != 'failed'),
                "domain_count": len(model_results['domain_info']),
                "models_trained": successful_models
            },
            "clustering_results": {
                "total_clusters": total_clusters,
                "noise_incidents": total_noise,
                "noise_percentage": (total_noise / total_incidents * 100) if total_incidents > 0 else 0
            },
            "config_snapshot": {
                "hdbscan_params": self.config.clustering.hdbscan,
                "domain_grouping": self.config.clustering.domain_grouping,
                "training_params": self.config.training.parameters
            }
        }
        
        return metadata
    
    async def _store_training_results(self, tech_center: str, model_results: Dict, 
                                    metadata: Dict, version: str) -> Dict:
        """Store training results in versioned BigQuery table"""
        try:
            # Generate model hash for versioning
            model_hash = self._generate_model_hash(model_results, metadata)
            
            # Get versioned table name
            versioned_table = self.config.bigquery.get_versioned_table_name(version, model_hash)
            
            # Prepare training results for storage
            storage_data = []
            for model_key, model_data in model_results['models'].items():
                if model_data.get('status') != 'failed':
                    # Create record for each clustered incident
                    incidents_data = model_data.get('incident_data', [])
                    labels = model_data['cluster_labels']
                    probabilities = model_data.get('probabilities', [])
                    
                    for i, (incident, label) in enumerate(zip(incidents_data, labels)):
                        record = {
                            "version": version,
                            "model_hash": model_hash,
                            "tech_center": tech_center,
                            "domain_name": model_data['domain_name'],
                            "incident_id": incident.get('incident_id'),
                            "cluster_label": int(label),
                            "cluster_probability": float(probabilities[i]) if i < len(probabilities) else None,
                            "is_noise": label == -1,
                            "training_date": datetime.now().isoformat(),
                            "model_metadata": json.dumps(metadata)
                        }
                        storage_data.append(record)
            
            # Store in BigQuery
            if storage_data:
                # Here you would use BigQuery client to insert data
                # await self.bigquery_client.insert_training_results(versioned_table, storage_data)
                logging.info("Would store %d training result records in table %s", 
                           len(storage_data), versioned_table)
                
                return {
                    "status": "success",
                    "table_name": versioned_table,
                    "records_stored": len(storage_data),
                    "model_hash": model_hash
                }
            else:
                return {"status": "failed", "reason": "no_data_to_store"}
                
        except Exception as e:
            logging.error("Failed to store training results: %s", str(e))
            return {"status": "failed", "error": str(e)}
    
    def _generate_model_hash(self, model_results: Dict, metadata: Dict) -> str:
        """Generate hash for model versioning"""
        hash_algorithm = self.config.training.versioning['hash_algorithm']
        hash_length = self.config.training.versioning['hash_length']
        
        # Create deterministic hash from model results and metadata
        hash_data = {
            "model_count": len(model_results['models']),
            "domain_info": model_results['domain_info'],
            "training_date": metadata['training_info']['training_date'][:10],  # Date only
            "config_hash": str(hash(str(metadata['config_snapshot'])))
        }
        
        hash_str = json.dumps(hash_data, sort_keys=True)
        hash_obj = hashlib.new(hash_algorithm)
        hash_obj.update(hash_str.encode('utf-8'))
        
        return hash_obj.hexdigest()[:hash_length]
    
    def get_training_statistics(self) -> Dict:
        """Get training pipeline statistics"""
        return self.training_stats
    
    async def validate_training_data(self, tech_center: str, 
                                   preprocessing_data: Dict) -> Dict:
        """Validate training data before training"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check embeddings
        embeddings = preprocessing_data.get('embeddings', [])
        if len(embeddings) == 0:
            validation_results["valid"] = False
            validation_results["errors"].append("No embeddings available")
            return validation_results
        
        # Check minimum incidents
        min_incidents = self.config.clustering.min_incidents_per_domain
        if len(embeddings) < min_incidents:
            validation_results["warnings"].append(
                f"Low incident count ({len(embeddings)} < {min_incidents})"
            )
        
        # Check embedding dimensions
        if len(embeddings) > 0:
            embedding_dim = len(embeddings[0]) if hasattr(embeddings[0], '__len__') else 0
            if embedding_dim == 0:
                validation_results["valid"] = False
                validation_results["errors"].append("Invalid embedding dimensions")
        
        return validation_results