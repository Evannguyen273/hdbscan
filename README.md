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
├── main.py                       # Enhanced CLI entry point
├── pipeline.py                   # Original modular pipeline
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

### 3. How to Run the Pipeline

#### **Option A: Enhanced Pipeline (main.py) - RECOMMENDED**

The enhanced pipeline supports tech centers, quarterly training, and Azure Functions architecture.

##### **Individual Pipeline Commands:**
```bash
# Preprocessing (detect new incidents, generate embeddings)
python main.py preprocess

# Training (quarterly model retraining for all tech centers)
python main.py train --year 2024 --quarter q4

# Prediction (classify incidents using trained models)
python main.py predict

# Check pipeline status
python main.py status
```

##### **Specific Tech Center Operations:**
```bash
# Process single tech center
python main.py preprocess --tech-center "BT-TC-Product Development & Engineering"

# Train single tech center
python main.py train --tech-center "BT-TC-Infrastructure Services" --year 2024 --quarter q1

# Predict for single tech center
python main.py predict --tech-center "BT-TC-Network Operations"
```

##### **Automated Scheduler (Production Mode):**
```bash
# Run continuous automated pipeline (recommended for production)
python main.py schedule
```
This runs:
- Preprocessing every 1 hour
- Prediction every 2 hours  
- Manual quarterly training

##### **Legacy Support:**
```bash
# Run original pipeline through enhanced interface
python main.py legacy \
  --query "SELECT * FROM incidents WHERE created_date >= '2024-01-01'" \
  --dataset "test_dataset" \
  --embeddings-table "project.dataset.embeddings" \
  --results-table "project.dataset.results"
```

#### **Option B: Original Modular Pipeline (pipeline.py)**

The original pipeline for traditional stage-by-stage execution without tech center support.

##### **Complete Pipeline:**
```bash
# Run full pipeline (all 4 stages)
python -c "
from pipeline import ClusteringPipeline
pipeline = ClusteringPipeline('config/config.yaml')
results = pipeline.run_modular_pipeline(
    input_query='SELECT * FROM your_table LIMIT 1000',
    embeddings_table_id='project.dataset.embeddings',
    results_table_id='project.dataset.results', 
    dataset_name='test_run'
)
print('Pipeline completed successfully!')
"
```

##### **Stage-by-Stage Execution:**
```bash
# Stage 1: Generate embeddings only
python -c "
from pipeline import ClusteringPipeline
pipeline = ClusteringPipeline()
results = pipeline.run_modular_pipeline(
    input_query='SELECT * FROM your_table',
    embeddings_table_id='project.dataset.embeddings',
    results_table_id=None,
    dataset_name='test_run',
    start_from_stage=1,
    end_at_stage=1
)
"

# Stage 2: Train HDBSCAN (using existing embeddings)
python -c "
from pipeline import ClusteringPipeline
pipeline = ClusteringPipeline()
results = pipeline.run_modular_pipeline(
    input_query='',
    embeddings_table_id=None,
    results_table_id=None,
    dataset_name='test_run',
    start_from_stage=2,
    end_at_stage=2
)
"

# Stage 3: Analyze clusters
python -c "
from pipeline import ClusteringPipeline
pipeline = ClusteringPipeline()
results = pipeline.run_modular_pipeline(
    input_query='',
    embeddings_table_id=None,
    results_table_id=None,
    dataset_name='test_run',
    start_from_stage=3,
    end_at_stage=3
)
"

# Stage 4: Save results to BigQuery
python -c "
from pipeline import ClusteringPipeline
pipeline = ClusteringPipeline()
results = pipeline.run_modular_pipeline(
    input_query='',
    embeddings_table_id=None,
    results_table_id='project.dataset.results',
    dataset_name='test_run',
    start_from_stage=4,
    end_at_stage=4
)
"
```

##### **Custom Parameters:**
```bash
# Run with custom parameters
python -c "
from pipeline import ClusteringPipeline
pipeline = ClusteringPipeline()
results = pipeline.run_modular_pipeline(
    input_query='SELECT number, short_description, description FROM incidents WHERE tech_center = \"BT-TC-Product Development\"',
    embeddings_table_id='project.dataset.embeddings',
    results_table_id='project.dataset.results',
    dataset_name='product_dev_analysis',
    summary_path='path/to/precomputed_summaries.parquet',
    write_disposition='WRITE_TRUNCATE',
    start_from_stage=1,
    end_at_stage=4,
    use_checkpoint=True
)
"
```

### 4. Which Pipeline to Choose?

| Use Case | Recommended Pipeline | Command |
|----------|---------------------|---------|
| **Production Deployment** | Enhanced (`main.py`) | `python main.py schedule` |
| **Tech Center Management** | Enhanced (`main.py`) | `python main.py train --tech-center "..."` |
| **Quarterly Training** | Enhanced (`main.py`) | `python main.py train --quarter q4` |
| **Azure Functions** | Enhanced (`main.py`) | Individual pipeline functions |
| **Research/Development** | Original (`pipeline.py`) | Stage-by-stage execution |
| **Custom Queries** | Original (`pipeline.py`) | Full parameter control |
| **Legacy Support** | Enhanced (`main.py`) | `python main.py legacy ...` |

### 5. Typical Workflows

#### **Production Workflow (Enhanced):**
```bash
# 1. Start automated scheduler
python main.py schedule

# 2. Quarterly training (manual)
python main.py train --year 2024 --quarter q4

# 3. Monitor status
python main.py status
```

#### **Development Workflow (Original):**
```bash
# 1. Test with small dataset
python -c "from pipeline import ClusteringPipeline; ..." 

# 2. Analyze specific stages
python -c "..." --start_from_stage=2 --end_at_stage=3

# 3. Full production run
python -c "..." --start_from_stage=1 --end_at_stage=4
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
-- Team services mapping
your-project.your_dataset.team_services_table

-- Raw incidents
your-project.your_dataset.incident_table

-- Problems data
your-project.your_dataset.problem_table

-- Preprocessed with embeddings  
your-project.preprocessing_pipeline.preprocessed_incidents

-- Prediction results
your-project.your_dataset.clustering_predictions

-- Watermark tracking
your-project.preprocessing_pipeline.preprocessing_watermarks
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

## ⚙️ Configuration

### 🔒 **Security-First Configuration Approach**

This pipeline uses a **template-based configuration system** to ensure no sensitive information is ever committed to the repository.

### Configuration Files Structure
- **`config/config_template.py`** - ✅ Safe template (committed to git)
- **`config/config_template.yaml`** - ✅ Safe template (committed to git)  
- **`config/config.py`** - ❌ Your actual config (ignored by git)
- **`config/config.yaml`** - ❌ Your actual config (ignored by git)
- **`.env.example`** - ✅ Safe template (committed to git)
- **`.env`** - ❌ Your actual credentials (ignored by git)

### 🚀 **Quick Setup Process**

1. **Copy Template Files**:
   ```bash
   # Copy configuration templates
   cp config/config_template.py config/config.py
   cp config/config_template.yaml config/config.yaml
   cp .env.example .env
   ```

2. **Fill in Your Actual Values**:
   ```bash
   # Edit with your real credentials (these files are gitignored)
   nano .env              # Add your actual environment variables
   nano config/config.yaml # Replace YOUR_PROJECT_ID with actual values
   ```

3. **Test Configuration**:
   ```bash
   # Verify everything works
   python test_config.py
   ```

### Environment Variables (.env file)
```bash
# Google Cloud Service Account Configuration  
SERVICE_ACCOUNT_KEY_PATH={"type": "service_account", "project_id": "your-project-id", "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"}

# BigQuery Table Configuration
TEAM_SERVICES_TABLE=your-project.your_dataset.team_services_table
INCIDENT_TABLE=your-project.your_dataset.incident_table
PROBLEM_TABLE=your-project.your_dataset.problem_table

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your-chat-deployment-name

# Azure OpenAI Embeddings Configuration
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-embedding-resource.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings
AZURE_OPENAI_EMBEDDING_API_VERSION=2023-05-15
AZURE_OPENAI_EMBEDDING_KEY=your_embedding_api_key_here
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Azure Blob Storage Configuration
BLOB_CONNECTION_STRING=https://yourstorageaccount.blob.core.windows.net/your-container
```

### Quick Start
1. **Setup Configuration**:
   ```bash
   # Copy template files to create your actual config
   cp config/config_template.py config/config.py
   cp config/config_template.yaml config/config.yaml
   cp .env.example .env
   
   # Edit with your actual credentials (these files are gitignored)
   nano .env
   nano config/config.yaml
   
   # Test configuration
   python test_config.py
   ```

2. **Validate Setup**:
   ```bash
   python main.py validate
   ```

3. **Check Status**:
   ```bash
   python main.py status
   ```

4. **Run Pipeline**:
   ```bash
   # Preprocess incidents
   python main.py preprocess --limit 100
   
   # Train models
   python main.py train --quarter q4
   
   # Run predictions  
   python main.py predict
   ```

// ...existing architecture and features sections...