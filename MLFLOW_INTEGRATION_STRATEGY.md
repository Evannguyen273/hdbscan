# MLflow Integration Strategy - Minimal Implementation

## 🎯 **Recommended Approach: MLflow for Tracking + Keep Existing Deployment**

### **Phase 1: Add MLflow Tracking (Week 1-2)**
```python
# training/clustering_trainer.py - Enhanced with MLflow
import mlflow
import mlflow.sklearn
from datetime import datetime

class ClusteringTrainer:
    def __init__(self, config):
        # ...existing code...
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.get('mlflow_tracking_uri', 'sqlite:///mlflow.db'))
        mlflow.set_experiment(f"HDBSCAN-Clustering-{datetime.now().year}")
    
    def run_complete_training(self, embedding_matrix, valid_indices, hdbscan_params, 
                            tech_center=None, quarter=None, **kwargs):
        
        # Start MLflow run
        run_name = f"{tech_center}-{quarter}" if tech_center and quarter else "training"
        
        with mlflow.start_run(run_name=run_name):
            # Log training parameters
            mlflow.log_params({
                "tech_center": tech_center,
                "quarter": quarter,
                "dataset_size": len(embedding_matrix),
                "embedding_dimensions": embedding_matrix.shape[1],
                **hdbscan_params
            })
            
            # ...existing training code...
            success, results = self._run_existing_training(embedding_matrix, valid_indices, hdbscan_params, **kwargs)
            
            if success:
                # Log performance metrics
                metrics = results["metrics_results"]["metrics"]
                mlflow.log_metrics({
                    "n_clusters": metrics.get("n_clusters", 0),
                    "silhouette_score": metrics.get("silhouette_score", 0),
                    "noise_ratio": metrics.get("noise_ratio", 0),
                    "training_duration": results.get("training_duration_seconds", 0),
                    "used_fallback": int(results["training_results"].get("used_fallback", False))
                })
                
                # Log model artifacts (keep existing save logic)
                if self.trained_model:
                    mlflow.sklearn.log_model(
                        self.trained_model, 
                        "hdbscan_model",
                        registered_model_name=f"HDBSCAN-{tech_center}" if tech_center else "HDBSCAN-Generic"
                    )
                
                # Log training metadata
                mlflow.log_dict(results["training_results"], "training_metadata.json")
                
                # Tag the run
                mlflow.set_tags({
                    "model_type": "HDBSCAN",
                    "pipeline_version": "v2.0",
                    "quality_status": "acceptable" if results["quality_results"]["acceptable"] else "warnings"
                })
            
            return success, results
```

### **Phase 2: MLflow Model Registry (Week 3-4)**
```python
# pipeline/training_pipeline.py - Enhanced with model registry
class TrainingPipeline:
    def train_tech_center_quarterly(self, tech_center: str, year: int, quarter: str):
        # ...existing training logic...
        
        if training_successful:
            # Register model in MLflow
            model_version = self._register_model_version(
                tech_center=tech_center,
                year=year,
                quarter=quarter,
                model_uri=mlflow.get_artifact_uri("hdbscan_model"),
                performance_metrics=training_results["metrics"]
            )
            
            # Keep existing Azure Blob storage logic
            self._save_to_azure_blob(model_artifacts, tech_center, year, quarter)
            
            return model_version
    
    def _register_model_version(self, tech_center, year, quarter, model_uri, performance_metrics):
        client = mlflow.tracking.MlflowClient()
        
        # Register new version
        model_version = client.create_model_version(
            name=f"HDBSCAN-{tech_center}",
            source=model_uri,
            description=f"Quarterly model for {tech_center} - {year}-{quarter}"
        )
        
        # Add version metadata
        client.set_model_version_tag(
            name=f"HDBSCAN-{tech_center}",
            version=model_version.version,
            key="quarter",
            value=f"{year}-{quarter}"
        )
        
        client.set_model_version_tag(
            name=f"HDBSCAN-{tech_center}",
            version=model_version.version,
            key="silhouette_score",
            value=str(performance_metrics.get("silhouette_score", 0))
        )
        
        # Auto-promote to staging for validation
        client.transition_model_version_stage(
            name=f"HDBSCAN-{tech_center}",
            version=model_version.version,
            stage="Staging"
        )
        
        return model_version
```

### **Phase 3: Enhanced CLI with MLflow (Week 5)**
```python
# main.py - Enhanced with MLflow queries
import mlflow
from mlflow.tracking import MlflowClient

def cmd_model_status(args):
    """Show model performance across quarters and tech centers"""
    client = MlflowClient()
    
    print("🎯 Model Performance Dashboard")
    print("=" * 60)
    
    for tech_center in TECH_CENTERS:
        try:
            model_name = f"HDBSCAN-{tech_center}"
            
            # Get latest production model
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
            
            print(f"\n📊 {tech_center}:")
            
            if prod_versions:
                prod_version = prod_versions[0]
                prod_run = client.get_run(prod_version.run_id)
                print(f"  Production: v{prod_version.version} (Q{prod_run.data.tags.get('quarter', 'Unknown')})")
                print(f"    Silhouette: {prod_run.data.metrics.get('silhouette_score', 'N/A'):.3f}")
                print(f"    Clusters: {prod_run.data.metrics.get('n_clusters', 'N/A')}")
            
            if staging_versions:
                staging_version = staging_versions[0] 
                staging_run = client.get_run(staging_version.run_id)
                print(f"  Staging: v{staging_version.version} (Q{staging_run.data.tags.get('quarter', 'Unknown')})")
                print(f"    Silhouette: {staging_run.data.metrics.get('silhouette_score', 'N/A'):.3f}")
                
        except Exception as e:
            print(f"  ❌ No models found: {str(e)}")

def cmd_model_compare(args):
    """Compare model performance across quarters"""
    client = MlflowClient()
    tech_center = args.tech_center
    
    print(f"📈 Performance Comparison: {tech_center}")
    print("=" * 60)
    
    # Get all versions for tech center
    model_name = f"HDBSCAN-{tech_center}"
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    performance_data = []
    for version in all_versions:
        run = client.get_run(version.run_id)
        performance_data.append({
            "version": version.version,
            "quarter": run.data.tags.get("quarter", "Unknown"),
            "silhouette": run.data.metrics.get("silhouette_score", 0),
            "clusters": run.data.metrics.get("n_clusters", 0),
            "noise_ratio": run.data.metrics.get("noise_ratio", 0),
            "stage": version.current_stage
        })
    
    # Sort by quarter and display
    performance_data.sort(key=lambda x: x["quarter"])
    
    print(f"{'Version':<8} {'Quarter':<10} {'Stage':<12} {'Silhouette':<12} {'Clusters':<10} {'Noise %':<8}")
    print("-" * 70)
    
    for data in performance_data:
        print(f"{data['version']:<8} {data['quarter']:<10} {data['stage']:<12} "
              f"{data['silhouette']:<12.3f} {data['clusters']:<10} {data['noise_ratio']:<8.1%}")

def cmd_model_promote(args):
    """Promote staging model to production"""
    client = MlflowClient()
    tech_center = args.tech_center
    
    model_name = f"HDBSCAN-{tech_center}"
    
    # Get staging model
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    
    if not staging_versions:
        print(f"❌ No staging model found for {tech_center}")
        return
    
    staging_version = staging_versions[0]
    
    # Get performance metrics
    run = client.get_run(staging_version.run_id)
    silhouette = run.data.metrics.get("silhouette_score", 0)
    
    print(f"🎯 Promoting {tech_center} model v{staging_version.version} to Production")
    print(f"   Quarter: {run.data.tags.get('quarter', 'Unknown')}")
    print(f"   Silhouette Score: {silhouette:.3f}")
    
    if silhouette >= 0.15:  # Quality threshold
        # Transition current production to archived
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for prod_version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=prod_version.version,
                stage="Archived"
            )
        
        # Promote staging to production
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Production"
        )
        
        print("✅ Model promoted to Production successfully")
    else:
        print(f"❌ Model quality too low (silhouette {silhouette:.3f} < 0.15)")

# Enhanced CLI commands
if __name__ == "__main__":
    parser.add_parser('model-status', help='Show model performance dashboard')
    parser.add_parser('model-compare', help='Compare model performance across quarters')
    parser.add_parser('model-promote', help='Promote staging model to production')
```

### **Configuration Addition**
```yaml
# config/enhanced_config.yaml
mlflow:
  tracking_uri: "sqlite:///mlflow.db"  # Local SQLite for development
  # tracking_uri: "azure://mlflow"     # Azure ML for production
  experiment_name: "HDBSCAN-Clustering"
  
  model_registry:
    enable_auto_registration: true
    quality_threshold: 0.15           # Minimum silhouette score for promotion
    
pipeline:
  enable_mlflow_tracking: true
  # ...existing config...
```

### **New CLI Commands Available**
```bash
# View model performance dashboard
python main.py model-status

# Compare quarters for specific tech center
python main.py model-compare --tech-center "BT-TC-Data Analytics"

# Promote staging model to production
python main.py model-promote --tech-center "BT-TC-Data Analytics"

# Training with automatic MLflow tracking
python main.py train --quarter q4  # Now automatically logs to MLflow
```

## 🤔 **MLflow vs Current Approach - Decision Matrix**

### **Comparative Analysis**

| Aspect | Current Approach | With MLflow | Recommendation |
|--------|------------------|-------------|----------------|
| **Model Versioning** | Manual quarterly folders | Automatic semantic versioning | ✅ **MLflow Better** |
| **Performance Tracking** | JSON files per model | Centralized metrics database | ✅ **MLflow Better** |
| **Model Comparison** | Manual file analysis | UI + API queries | ✅ **MLflow Better** |
| **Deployment** | GitHub Actions + Blob | Keep existing + MLflow registry | ⚖️ **Hybrid Best** |
| **Complexity** | Simple folder structure | Additional MLflow setup | ⚠️ **Current Simpler** |
| **Team Learning Curve** | Zero (already working) | Medium (MLflow concepts) | ⚠️ **Current Easier** |
| **Rollback Capability** | Manual symlink management | Automatic model promotion | ✅ **MLflow Better** |
| **Experiment Tracking** | Not available | Full parameter/metric history | ✅ **MLflow Better** |
| **Cost** | Just storage costs | Storage + MLflow hosting | ⚠️ **Current Cheaper** |

### **ROI Analysis**

#### **Current Pain Points (MLflow Solves)**
1. **❓ "Which quarter had best performance?"** → MLflow: `SELECT * FROM metrics WHERE metric='silhouette_score' ORDER BY value DESC`
2. **❓ "Can I safely rollback Q4 model?"** → MLflow: One-click stage transition 
3. **❓ "Which parameters work best?"** → MLflow: Parameter correlation analysis
4. **❓ "Is model performance degrading?"** → MLflow: Automated drift detection

#### **Time Savings Calculation**
```
Quarterly Model Analysis (Current): 
- Manual metric collection: 2 hours
- Performance comparison: 1 hour  
- Documentation updates: 1 hour
- Total: 4 hours × 4 quarters = 16 hours/year

With MLflow:
- Automatic metric collection: 0 hours
- Performance comparison: 10 minutes
- Documentation: Auto-generated
- Total: 0.5 hours × 4 quarters = 2 hours/year

Annual Time Savings: 14 hours
```

## 💡 **Final Recommendation**

### **🎯 Recommended: Hybrid Approach**

**Keep your existing deployment pipeline + Add MLflow for tracking only**

#### **What to Keep (Works Great)**
- ✅ GitHub Actions for CI/CD
- ✅ Azure Blob Storage for model artifacts
- ✅ Function Apps + VM hybrid architecture
- ✅ Quarterly training schedule
- ✅ Training guardrails system

#### **What to Add (High Value)**
- 🆕 MLflow tracking for experiment metrics
- 🆕 MLflow model registry for versioning
- 🆕 Performance comparison dashboards
- 🆕 Automated model promotion workflow

#### **Implementation Roadmap**

##### **Phase 1: MLflow Tracking (Low Risk, High Value)**
```python
# Minimal change - just add tracking to existing training
with mlflow.start_run():
    mlflow.log_params(hdbscan_params)
    # ...existing training code...
    mlflow.log_metrics(performance_metrics)
    mlflow.sklearn.log_model(model, "hdbscan_model")
```

##### **Phase 2: Model Registry (Medium Risk, High Value)**
```python
# Add model versioning and stage management
mlflow.register_model(model_uri, f"HDBSCAN-{tech_center}")
# Keep existing blob storage as backup
```

##### **Phase 3: Enhanced CLI (Low Risk, Medium Value)**
```bash
# New commands for model management
python main.py model-status
python main.py model-compare --tech-center "..."
python main.py model-promote --tech-center "..."
```

#### **Risk Mitigation**
- 🛡️ **Dual Storage**: Continue saving to Azure Blob as primary, MLflow as secondary
- 🛡️ **Gradual Rollout**: Start with 1-2 tech centers, expand gradually
- 🛡️ **Fallback Ready**: Can disable MLflow anytime without breaking existing pipeline

## 📊 **Deployment Options**

### **Option A: SQLite MLflow (Development/Small Teams)**
```yaml
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  cost: $0/month
  setup_time: 10 minutes
  scalability: Single user
```

### **Option B: Azure ML MLflow (Production/Large Teams)**
```yaml
mlflow:
  tracking_uri: "azureml://..."
  cost: ~$50-100/month
  setup_time: 2-3 hours
  scalability: Multi-user, enterprise
```

### **Option C: Self-Hosted MLflow (Balanced)**
```yaml
mlflow:
  tracking_uri: "postgresql://..."
  cost: ~$20-50/month (VM + DB)
  setup_time: 1 day
  scalability: Team-sized
```

## 🎯 **Quick Decision Framework**

### **Choose MLflow IF:**
- ✅ You want better quarterly performance analysis
- ✅ You need model rollback capabilities
- ✅ Team wants to experiment with parameters
- ✅ You have 2+ weeks for implementation
- ✅ Budget allows $50-100/month for tracking

### **Stick with Current IF:**
- ✅ Current system meets all needs
- ✅ Team prefers simple folder structure
- ✅ No time for additional tooling
- ✅ Want to minimize dependencies
- ✅ Cost optimization is priority

## 🚀 **My Recommendation for Your H&M Pipeline**

### **Start with Phase 1: Minimal MLflow Integration**

```python
# Week 1: Add 10 lines to existing training
import mlflow

class ClusteringTrainer:
    def run_complete_training(self, ...):
        with mlflow.start_run():
            mlflow.log_params(hdbscan_params)
            # ...existing code unchanged...
            mlflow.log_metrics({"silhouette_score": score})
            mlflow.sklearn.log_model(self.trained_model, "model")
        
        # Keep all existing save logic unchanged
        return success, results
```

**Benefits Immediately Available:**
- 📊 Visual performance tracking across quarters
- 🔍 Parameter correlation analysis  
- 📈 Automated performance dashboards
- 🎯 Zero risk to existing pipeline

**After 1 quarter of use, evaluate:**
- Did the team find MLflow valuable?
- Are the dashboards being used?
- Is model comparison helping decisions?

**If yes → Proceed to Phase 2 (Model Registry)**
**If no → Remove MLflow (10 lines of code to delete)**

This gives you the **best of both worlds** with minimal risk and maximum learning opportunity! 🎯