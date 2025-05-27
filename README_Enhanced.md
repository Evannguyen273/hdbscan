# Enhanced HDBSCAN Clustering Pipeline for Tech Centers

A modular, scalable pipeline for incident clustering across multiple tech centers with quarterly model retraining and real-time prediction capabilities.

## 🏗️ Architecture Overview

```
clustering_pipeline/
├── config/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   └── enhanced_config.yaml      # Enhanced configuration
├── data/
│   ├── __init__.py
│   ├── bigquery_client.py        # BigQuery operations
│   └── blob_storage.py           # Azure Blob Storage operations
├── preprocessing/
│   ├── __init__.py
│   ├── text_processing.py        # Text cleaning, summarization
│   └── embedding_generation.py   # Semantic embedding creation (weights: semantic=1.0, entity=0.0, action=0.0)
├── core/
│   ├── __init__.py
│   └── clustering.py             # HDBSCAN training with UMAP
├── analysis/
│   ├── __init__.py
│   ├── cluster_analysis.py       # Cluster analysis & metrics
│   ├── cluster_labeling.py       # LLM-based intelligent labeling
│   └── domain_grouping.py        # Hierarchical domain grouping
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing_pipeline.py # Hourly preprocessing pipeline
│   ├── training_pipeline.py      # Quarterly training pipeline
│   ├── prediction_pipeline.py    # 2-hourly prediction pipeline
│   └── orchestrator.py           # Pipeline coordination
├── utils/
│   ├── __init__.py
│   └── file_utils.py             # File operation utilities
├── main_enhanced.py              # Enhanced CLI entry point
├── pipeline.py                   # Legacy pipeline (original)
└── requirements_enhanced.txt     # All dependencies
```

## ✨ Key Features

### 🔄 **Multi-Pipeline Architecture**
- **Preprocessing Pipeline**: Runs every hour to detect new incidents and generate embeddings
- **Training Pipeline**: Quarterly retraining for all 15 tech centers (parallel execution)
- **Prediction Pipeline**: Real-time classification every 2 hours
- **Orchestrator**: Automated scheduling and coordination

### 🏢 **Tech Center Support**
- **15 Tech Centers**: Individual model training per tech center
- **Parallel Training**: Configurable concurrent model training
- **Model Artifacts**: Organized storage structure per tech center/quarter
- **Watermark Tracking**: Prevents duplicate processing using `sys_created_on`

### 📊 **Quarterly Model Management**
- **Q1, Q2, Q3, Q4**: Automatic quarter detection and training
- **Model Versioning**: Quarterly snapshots with metadata
- **Artifact Storage**: Local + Azure Blob Storage
- **Current Model Links**: Symlinks to latest trained models

### 🎯 **Semantic-Only Embeddings**
- **Entity Weight**: 0.0 (disabled as requested)
- **Action Weight**: 0.0 (disabled as requested)
- **Semantic Weight**: 1.0 (pure semantic embeddings)

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements_enhanced.txt

# Set environment variables
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-key"
export BLOB_CONNECTION_STRING="your-connection-string"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

### 2. Configuration
Edit `config/enhanced_config.yaml`:
```yaml
clustering:
  embedding:
    weights:
      entity: 0.0      # Disabled
      action: 0.0      # Disabled  
      semantic: 1.0    # Pure semantic

pipeline:
  save_to_local: true
  result_path: "/your/result/path"
  parallel_training: true
  max_workers: 4
```

### 3. Run Pipelines

#### Preprocessing (Hourly)
```bash
# All tech centers
python main_enhanced.py preprocess

# Specific tech center
python main_enhanced.py preprocess --tech-center "BT-TC-Product Development & Engineering"
```

#### Training (Quarterly)
```bash
# All tech centers (parallel)
python main_enhanced.py train

# Specific tech center
python main_enhanced.py train --tech-center "BT-TC-Infrastructure Services" --year 2024 --quarter q1

# Current quarter
python main_enhanced.py train --year 2024 --quarter q3
```

#### Prediction (Every 2 hours)
```bash
# All tech centers
python main_enhanced.py predict

# Specific tech center
python main_enhanced.py predict --tech-center "BT-TC-Network Operations"
```

#### Automated Scheduler
```bash
# Run continuous scheduler
python main_enhanced.py schedule
```

#### Status Check
```bash
python main_enhanced.py status
```

## 📁 Blob Storage Structure

```
prediction-artifacts/
├── models/
│   ├── BT_TC_Product_Development_Engineering/
│   │   ├── 2024/
│   │   │   ├── q1/
│   │   │   │   ├── clustering/
│   │   │   │   │   ├── hdbscan_clusterer.pkl
│   │   │   │   │   └── umap_reducer.pkl
│   │   │   │   ├── analysis/
│   │   │   │   │   ├── labeled_clusters.json
│   │   │   │   │   └── domains.json
│   │   │   │   └── metadata/
│   │   │   │       └── training_metadata.json
│   │   │   ├── q2/ # Similar structure
│   │   │   ├── q3/
│   │   │   └── q4/
│   │   └── current -> 2024/q3  # Symlink to current quarter
│   └── {other_tech_centers}/...
├── predictions/
│   ├── BT_TC_Product_Development_Engineering/
│   │   └── 2024/q3/
│   │       ├── batch_predictions/
│   │       └── real_time_predictions/
└── monitoring/
    ├── drift_detection/
    ├── performance_tracking/
    └── alerts/
```

## 🗄️ Database Tables

### BigQuery Tables
```sql
-- Raw incidents
enterprise-dashboardnp-cd35.bigquery_datasets_spoke_oa_dev.incidents_table

-- Preprocessed with embeddings  
enterprise-dashboardnp-cd35.preprocessing_pipeline.preprocessed_incidents

-- Prediction results
enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.clustering_predictions

-- Watermark tracking
enterprise-dashboardnp-cd35.preprocessing_pipeline.preprocessing_watermarks
```

### Key Columns
- `number`: Incident identifier
- `sys_created_on`: Creation timestamp (for watermarking)
- `combined_incidents_summary`: AI-generated summary
- `embedding`: Semantic embedding vector
- `tech_center`: Tech center assignment

## 🔄 Pipeline Schedules

| Pipeline | Frequency | Purpose |
|----------|-----------|---------|
| **Preprocessing** | Every 1 hour | Detect new incidents, generate summaries/embeddings |
| **Prediction** | Every 2 hours | Classify new incidents using trained models |
| **Training** | Quarterly | Retrain models for data drift (Q1, Q2, Q3, Q4) |

## 🏢 Tech Centers (15 Total)

1. BT-TC-Product Development & Engineering
2. BT-TC-Infrastructure Services  
3. BT-TC-Network Operations
4. BT-TC-Security Operations
5. BT-TC-Cloud Services
6. BT-TC-Data Analytics
7. BT-TC-Enterprise Applications
8. BT-TC-Field Services
9. BT-TC-Customer Support
10. BT-TC-Quality Assurance
11. BT-TC-DevOps Engineering
12. BT-TC-Business Intelligence
13. BT-TC-Systems Integration
14. BT-TC-Mobile Solutions
15. BT-TC-Compliance & Governance

## 🎛️ Command Reference

### Preprocessing Commands
```bash
# Process all tech centers
python main_enhanced.py preprocess

# Process specific tech center
python main_enhanced.py preprocess --tech-center "BT-TC-Data Analytics"
```

### Training Commands
```bash
# Train all (current quarter)
python main_enhanced.py train

# Train all (specific quarter)
python main_enhanced.py train --year 2024 --quarter q2

# Train single tech center
python main_enhanced.py train --tech-center "BT-TC-Cloud Services" --year 2024 --quarter q1
```

### Prediction Commands
```bash
# Predict all tech centers
python main_enhanced.py predict

# Predict single tech center
python main_enhanced.py predict --tech-center "BT-TC-Security Operations"
```

### Legacy Support
```bash
# Run original pipeline
python main_enhanced.py legacy \
  --query "SELECT * FROM incidents WHERE created_date >= '2024-01-01'" \
  --dataset "test_dataset" \
  --embeddings-table "project.dataset.embeddings" \
  --results-table "project.dataset.results"
```

## 🔧 Configuration Options

### Key Settings
```yaml
clustering:
  hdbscan:
    min_cluster_size: 25    # Minimum cluster size
    min_samples: 5          # Minimum samples for core point
  umap:
    n_components: 50        # UMAP dimensions
    n_neighbors: 100        # UMAP neighbors
  embedding:
    weights:
      entity: 0.0           # Entity embedding weight (DISABLED)
      action: 0.0           # Action embedding weight (DISABLED)
      semantic: 1.0         # Semantic embedding weight (PURE SEMANTIC)

pipeline:
  save_to_local: true       # Enable local file saving
  parallel_training: true   # Enable parallel tech center training
  max_workers: 4           # Maximum concurrent workers
  
  preprocessing:
    frequency_minutes: 60   # Run every hour
    batch_size: 1000       # Batch size for processing
  
  prediction:
    frequency_minutes: 120  # Run every 2 hours
    batch_size: 500        # Batch size for prediction
```

## 🚦 Best Practices

### **Recommended Architecture**

1. **Separate Preprocessing Pipeline**: 
   - ✅ Runs every hour to detect new incidents
   - ✅ Generates embeddings on-the-go
   - ✅ Stores in preprocessing table for training/prediction

2. **Parallel Tech Center Training**:
   - ✅ Train all 15 tech centers simultaneously (quarterly)
   - ✅ Individual model artifacts per tech center
   - ✅ Configurable worker pool (max_workers: 4)

3. **Watermark-Based Processing**:
   - ✅ Uses `sys_created_on` to track processed incidents
   - ✅ Prevents duplicate processing
   - ✅ Efficient incremental updates

## 🚀 **Azure Function App Hybrid Framework (Recommended)**

### **Hybrid Architecture: Functions + VM**

**Perfect for production deployment with cost optimization and operational simplicity.**

#### **Azure Function Apps (Serverless - Automated 24/7)**

##### **Preprocessing Function App**
- **Schedule**: Timer Trigger every 1 hour (`0 0 * * * *`)
- **Duration**: 5-15 minutes per execution
- **Plan**: Consumption Plan (pay-per-execution)
- **Memory**: 512MB - 1GB
- **Cost**: ~$20-50/month

**What it does:**
```python
def main(timer: func.TimerRequest):
    pipeline = PreprocessingPipeline(config)
    result = pipeline.run_preprocessing_all_tech_centers()
    return result
```

##### **Prediction Function App**
- **Schedule**: Timer Trigger every 2 hours (`0 0 */2 * * *`)
- **Duration**: 10-20 minutes per execution  
- **Plan**: Premium P1V2 (for better performance)
- **Memory**: 2-4GB
- **Cost**: ~$100-200/month

**What it does:**
```python
def main(timer: func.TimerRequest):
    pipeline = PredictionPipeline(config)
    result = pipeline.run_predictions_all_tech_centers()
    return result
```

#### **Azure Virtual Machine (Manual - Quarterly)**

##### **Training Pipeline on VM**
- **Schedule**: Manual trigger (quarterly by data science team)
- **Duration**: 2-4 hours for all 15 tech centers
- **VM Size**: Standard_D8s_v3 (8 cores, 32GB RAM)
- **Usage**: 4 times/year × 4 hours = 16 hours total
- **Cost**: ~$50-100/quarter (pay only when running)

**What it does:**
```bash
# Quarterly training by data scientist
python main_enhanced.py train --year 2024 --quarter q4
python main_enhanced.py status
```

### **Why This Hybrid Approach is Optimal**

#### **✅ Cost Optimization**
- **Functions**: Auto-scaling serverless for routine operations
- **VM**: Pay only during quarterly training (16 hours/year)
- **Total Cost**: ~$200-400/month vs $2000+/month for always-on VMs

#### **✅ Operational Excellence**
- **Functions**: Zero maintenance, Azure-managed
- **Training**: Full control and monitoring for complex ML workloads
- **No timeout issues** for intensive training processes

#### **✅ Reliability & Performance**
- **Functions**: Native Azure scaling and availability
- **Training**: Dedicated resources when needed
- **Independent failure domains**

### **Resource Allocation Strategy**

#### **Function Apps (24/7 Automated)**
```yaml
func-preprocessing:
  plan: Consumption
  memory: 512MB
  timeout: 10 minutes
  schedule: "0 0 * * * *"  # Every hour
  
func-prediction:
  plan: Premium P1V2
  memory: 3.5GB
  timeout: 30 minutes  
  schedule: "0 0 */2 * * *"  # Every 2 hours
```

#### **Virtual Machine (Quarterly Manual)**
```yaml
vm-training:
  size: Standard_D8s_v3
  cores: 8
  memory: 32GB
  usage: 4 times/year × 4 hours
  access: RDP/SSH for data scientists
```

### **Implementation Phases**

#### **Phase 1: Function Apps (Week 1-2)**
1. Extract preprocessing logic → Function App
2. Extract prediction logic → Function App
3. Configure Timer Triggers
4. Test automated execution

#### **Phase 2: VM Training (Week 3)**
1. Configure dedicated training VM
2. Install full pipeline with enhanced CLI
3. Test quarterly training workflow
4. Document procedures for data science team

#### **Phase 3: Integration (Week 4)**
1. Ensure Functions can read VM-trained models from Blob Storage
2. Test complete quarterly cycle end-to-end
3. Set up monitoring and alerting
4. Train team on new hybrid workflow

### **Team Responsibilities**

#### **Automated (Zero Human Intervention)**
- ✅ Hourly incident preprocessing
- ✅ 2-hourly prediction classification
- ✅ Model loading from Blob Storage
- ✅ Error handling and automatic retries

#### **Manual (Data Science Team - Quarterly)**
- 🎯 Model retraining and validation
- 🎯 Parameter tuning and optimization  
- 🎯 Performance analysis and drift detection
- 🎯 Model artifact management

### **Daily Operations Flow**
```
Hour 0: [Function] Preprocessing → New incidents processed
Hour 2: [Function] Prediction → Classifications generated
Hour 3: [Function] Preprocessing → More incidents processed  
Hour 4: [Function] Prediction → More classifications generated
...continues automatically 24/7
```

### **Quarterly Operations Flow**
```
Quarter End:
1. Data Scientist connects to Training VM
2. Executes: python main_enhanced.py train --year 2024 --quarter q4
3. Monitors parallel training progress (2-4 hours)
4. Validates model performance metrics
5. Updates model artifacts in Blob Storage
6. Function Apps automatically use new models
```

### **Monitoring & Alerting**

#### **Function Apps**
- **Application Insights**: Performance metrics and error tracking
- **Alerts**: Execution failures or performance degradation
- **Dashboards**: Real-time processing volumes and success rates

#### **Training VM**
- **Manual Monitoring**: During quarterly training sessions
- **Email Notifications**: Training completion and performance reports
- **Model Metrics**: Quarterly performance comparison tracking

This hybrid approach delivers **enterprise-grade reliability** with **optimal cost efficiency** and **operational simplicity**!

### **Deployment Strategy**

```bash
# Production deployment
1. Setup automated scheduler
   python main_enhanced.py schedule

2. Manual quarterly training  
   python main_enhanced.py train --year 2024 --quarter q4

3. Monitor pipeline status
   python main_enhanced.py status
```

## 📊 Monitoring & Metrics

### Pipeline Metrics
- **Preprocessing**: Incidents processed per hour
- **Training**: Model performance per tech center  
- **Prediction**: Classification confidence scores
- **Drift Detection**: Quarterly model comparison

### Output Files
```
results/
├── preprocessing/logs/          # Preprocessing run logs
├── training/logs/              # Training run logs  
├── predictions/logs/           # Prediction run logs
└── models/{tech_center}/       # Model artifacts per tech center
```

## 🔒 Security & Compliance

- **Environment Variables**: Secure credential management
- **Azure Key Vault**: Integration ready for secrets
- **RBAC**: Role-based access to BigQuery/Blob Storage
- **Audit Logs**: Complete pipeline execution tracking

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Pipeline Status**: ✅ Production Ready  
**Tech Centers**: 15 Supported  
**Training Schedule**: Quarterly (Q1, Q2, Q3, Q4)  
**Embedding Type**: Pure Semantic (weights: semantic=1.0, entity=0.0, action=0.0)