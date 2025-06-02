# DEPLOYMENT GUIDE

## 🚀 **Production Deployment Guide**

Complete guide for deploying the HDBSCAN clustering pipeline with cumulative training and versioned model storage.

## 📋 **Pre-Deployment Checklist**

### **1. Infrastructure Requirements**
- ✅ **Google Cloud Project** with BigQuery enabled
- ✅ **Azure Storage Account** for blob storage
- ✅ **Azure OpenAI Service** for embeddings
- ✅ **Compute Instance** (VM/Azure Functions)
- ✅ **Network connectivity** between all services

### **2. Access & Permissions**
- ✅ **BigQuery Admin** role for data operations
- ✅ **Storage Blob Data Contributor** for Azure Blob
- ✅ **OpenAI API access** for embedding generation
- ✅ **Service account keys** properly configured

### **3. Storage Setup**
- ✅ **BigQuery dataset** created with proper permissions
- ✅ **Azure Blob container** `hdbscan-models` created
- ✅ **Storage cost budgets** configured
- ✅ **Data retention policies** implemented

## 🔧 **Step-by-Step Deployment**

### **Step 1: Environment Setup**

#### **Clone Repository**
```bash
git clone <your-repo-url>
cd hdbscan
```

#### **Install Dependencies**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

#### **Configure Environment Variables**
```bash
# Copy template
cp .env.example .env

# Edit with your actual values
nano .env
```

```bash
# .env file contents
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_EMBEDDING_KEY=your_embedding_api_key_here
BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### **Step 2: Configuration**

#### **Copy Configuration Templates**
```bash
cp config/config_template.yaml config/enhanced_config.yaml
```

#### **Edit Configuration**
```yaml
# config/enhanced_config.yaml
bigquery:
  project_id: "your-gcp-project"
  dataset_id: "hdbscan_pipeline"
  
blob_storage:
  container_name: "hdbscan-models"
  connection_string: "${BLOB_CONNECTION_STRING}"
  
training:
  approach: "cumulative"
  window_months: 24
  schedule: "semi_annual"
  
tech_centers:
  - "BT-TC-Data Analytics"
  - "BT-TC-Network Operations"
  - "BT-TC-Security Operations"
  # ... add all 15 tech centers
```

### **Step 3: Database Setup**

#### **Create BigQuery Dataset**
```bash
# Using gcloud CLI
gcloud config set project your-gcp-project
bq mk --dataset your-gcp-project:hdbscan_pipeline

# Or via Console: https://console.cloud.google.com/bigquery
```

#### **Create Required Tables**
```sql
-- preprocessed_incidents table
CREATE TABLE `your-gcp-project.hdbscan_pipeline.preprocessed_incidents` (
  number STRING,
  sys_created_on TIMESTAMP,
  combined_incidents_summary STRING,
  embedding ARRAY<FLOAT64>,
  tech_center STRING
);

-- incident_predictions table  
CREATE TABLE `your-gcp-project.hdbscan_pipeline.incident_predictions` (
  incident_id STRING,
  tech_center STRING,
  predicted_cluster_id INT64,
  predicted_cluster_label STRING,
  predicted_domain_id INT64,
  predicted_domain_name STRING,
  confidence_score FLOAT64,
  prediction_timestamp TIMESTAMP,
  model_table_used STRING,
  blob_model_path STRING
);
```

### **Step 4: Azure Blob Storage Setup**

#### **Create Container**
```bash
# Using Azure CLI
az storage container create \
  --name hdbscan-models \
  --connection-string "$BLOB_CONNECTION_STRING"
```

#### **Set Up Directory Structure**
```bash
# Directory structure will be created automatically:
# hdbscan-models/
# ├── bt-tc-data-analytics/
# │   ├── 2024_q4/
# │   └── 2025_q2/
# ├── bt-tc-network-operations/
# └── ...
```

### **Step 5: Initial Validation**

#### **Test Configuration**
```bash
python main.py validate
```

#### **Test Connectivity**
```bash
# Test BigQuery connection
python -c "
from config.config import Config
from google.cloud import bigquery
config = Config()
client = bigquery.Client(project=config.bigquery['project_id'])
print('BigQuery connection successful!')
"

# Test Azure Blob connection
python -c "
from azure.storage.blob import BlobServiceClient
import os
client = BlobServiceClient.from_connection_string(os.getenv('BLOB_CONNECTION_STRING'))
print('Blob storage connection successful!')
"
```

#### **Test OpenAI API**
```bash
python -c "
from embedding_service import EmbeddingService
service = EmbeddingService()
result = service.generate_embedding('Test incident summary')
print(f'Embedding generated: {len(result)} dimensions')
"
```

### **Step 6: Data Pipeline Setup**

#### **Initial Data Load**
```bash
# Process historical data (start with small batch)
python main.py preprocess --limit 1000

# Check results
python -c "
from google.cloud import bigquery
client = bigquery.Client()
query = 'SELECT COUNT(*) as count FROM \`your-gcp-project.hdbscan_pipeline.preprocessed_incidents\`'
result = list(client.query(query))
print(f'Preprocessed incidents: {result[0].count}')
"
```

#### **Initial Model Training**
```bash
# Train models for current quarter
python main.py train --year 2025 --quarter q2 --tech-centers "BT-TC-Data Analytics"

# Verify model artifacts in blob storage
python -c "
from azure.storage.blob import BlobServiceClient
import os
client = BlobServiceClient.from_connection_string(os.getenv('BLOB_CONNECTION_STRING'))
container_client = client.get_container_client('hdbscan-models')
blobs = list(container_client.list_blobs(name_starts_with='bt-tc-data-analytics/2025_q2/'))
print(f'Model artifacts: {len(blobs)} files uploaded')
for blob in blobs:
    print(f'  - {blob.name}')
"
```

#### **Test Predictions**
```bash
# Run test predictions
python main.py predict --tech-center "BT-TC-Data Analytics"

# Check prediction results
python -c "
from google.cloud import bigquery
client = bigquery.Client()
query = 'SELECT COUNT(*) as count FROM \`your-gcp-project.hdbscan_pipeline.incident_predictions\`'
result = list(client.query(query))
print(f'Predictions generated: {result[0].count}')
"
```

## 🔄 **Production Deployment Options**

### **Option A: VM Deployment (Recommended for Start)**

#### **1. Provision VM**
```bash
# Google Cloud VM
gcloud compute instances create hdbscan-pipeline \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=n1-standard-4 \
  --zone=us-central1-a
```

#### **2. Deploy Code**
```bash
# Copy code to VM
gcloud compute scp --recurse . hdbscan-pipeline:~/hdbscan/

# SSH into VM
gcloud compute ssh hdbscan-pipeline

# Setup environment
cd hdbscan
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **3. Setup Cron Jobs**
```bash
# Edit crontab
crontab -e

# Add scheduled jobs
# Preprocessing every hour
0 * * * * cd /home/user/hdbscan && /home/user/hdbscan/venv/bin/python main.py preprocess

# Predictions every 2 hours
0 */2 * * * cd /home/user/hdbscan && /home/user/hdbscan/venv/bin/python main.py predict

# Training on 1st of June and December (semi-annual)
0 0 1 6,12 * cd /home/user/hdbscan && /home/user/hdbscan/venv/bin/python training_orchestrator.py
```

#### **4. Setup Monitoring**
```bash
# Create log monitoring
sudo nano /etc/rsyslog.d/hdbscan.conf

# Add:
# :programname, isequal, "hdbscan" /var/log/hdbscan.log

# Restart rsyslog
sudo systemctl restart rsyslog
```

### **Option B: Azure Functions (Serverless)**

#### **1. Function App Structure**
```
hdbscan-functions/
├── preprocess-incidents/
│   ├── __init__.py
│   └── function.json
├── train-models/
│   ├── __init__.py
│   └── function.json
├── predict-incidents/
│   ├── __init__.py
│   └── function.json
└── requirements.txt
```

#### **2. Deploy Functions**
```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Initialize function app
func init hdbscan-functions --python

# Create functions
cd hdbscan-functions
func new --name preprocess-incidents --template "Timer trigger"
func new --name predict-incidents --template "Timer trigger"
func new --name train-models --template "Timer trigger"

# Deploy
func azure functionapp publish your-function-app-name
```

### **Option C: Kubernetes (Enterprise Scale)**

#### **1. Create Deployment**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hdbscan-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hdbscan-pipeline
  template:
    metadata:
      labels:
        app: hdbscan-pipeline
    spec:
      containers:
      - name: hdbscan
        image: your-registry/hdbscan:latest
        env:
        - name: AZURE_OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: hdbscan-secrets
              key: openai-key
```

#### **2. Create CronJobs**
```yaml
# k8s/cronjobs.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: hdbscan-preprocess
spec:
  schedule: "0 * * * *"  # Every hour
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: preprocess
            image: your-registry/hdbscan:latest
            command: ["python", "main.py", "preprocess"]
```

## 📊 **Monitoring & Alerting**

### **1. Log Monitoring**
```bash
# Centralized logging
tail -f logs/pipeline_$(date +%Y%m%d).log

# Error monitoring
grep -i error logs/pipeline_$(date +%Y%m%d).log | tail -10
```

### **2. Performance Monitoring**
```python
# monitoring/health_check.py
import psutil
import logging
from datetime import datetime

def check_system_health():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    
    if cpu_percent > 80:
        logging.warning(f"High CPU usage: {cpu_percent}%")
    if memory_percent > 80:
        logging.warning(f"High memory usage: {memory_percent}%")
    if disk_usage > 80:
        logging.warning(f"High disk usage: {disk_usage}%")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "disk_usage": disk_usage,
        "status": "healthy" if all([cpu_percent < 80, memory_percent < 80, disk_usage < 80]) else "warning"
    }
```

### **3. Data Quality Monitoring**
```python
# monitoring/data_quality.py
def check_data_quality():
    # Check for recent data
    query = """
    SELECT 
        COUNT(*) as recent_incidents,
        MAX(sys_created_on) as latest_incident
    FROM `your-project.hdbscan_pipeline.preprocessed_incidents`
    WHERE sys_created_on >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 HOUR)
    """
    # Implementation...
```

### **4. Alert Configuration**
```yaml
# alerts/config.yaml
alerts:
  email:
    enabled: true
    recipients:
      - "ops-team@company.com"
      - "data-team@company.com"
  
  thresholds:
    prediction_errors: 5  # Alert if >5 prediction errors in hour
    training_failures: 1  # Alert immediately on training failure
    data_lag_hours: 4     # Alert if no new data for 4 hours
    
  channels:
    slack:
      webhook_url: "https://hooks.slack.com/services/..."
      channel: "#data-alerts"
```

## 🔒 **Security & Compliance**

### **1. Secret Management**
```bash
# Use Azure Key Vault or Google Secret Manager
# Never store secrets in code or config files

# Example: Azure Key Vault
az keyvault secret set \
  --vault-name "hdbscan-secrets" \
  --name "openai-api-key" \
  --value "your-secret-key"
```

### **2. Access Control**
```bash
# Principle of least privilege
# Service account permissions:
# - BigQuery: datasets.get, tables.create, tables.get, tables.list
# - Storage: Storage Blob Data Contributor
# - OpenAI: API access only
```

### **3. Data Privacy**
```python
# Implement data masking for sensitive fields
def mask_sensitive_data(text):
    import re
    # Mask email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Mask phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    return text
```

## 📈 **Performance Optimization**

### **1. Batch Processing**
```python
# config/enhanced_config.yaml
pipeline:
  preprocessing:
    batch_size: 1000      # Process 1000 incidents at a time
  prediction:
    batch_size: 500       # Predict 500 incidents at a time
  training:
    max_workers: 4        # Parallel processing
```

### **2. Caching**
```python
# Use Redis for caching embeddings
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_embedding(text_hash):
    return r.get(f"embedding:{text_hash}")

def cache_embedding(text_hash, embedding):
    r.setex(f"embedding:{text_hash}", 3600, embedding)  # 1 hour TTL
```

### **3. Database Optimization**
```sql
-- Create indexes for performance
CREATE INDEX idx_preprocessed_incidents_created_on 
ON `your-project.hdbscan_pipeline.preprocessed_incidents`(sys_created_on);

CREATE INDEX idx_preprocessed_incidents_tech_center 
ON `your-project.hdbscan_pipeline.preprocessed_incidents`(tech_center);
```

## 🔄 **Backup & Recovery**

### **1. Data Backup Strategy**
```bash
# Automated BigQuery exports
bq extract \
  --destination_format=PARQUET \
  --compression=GZIP \
  'your-project:hdbscan_pipeline.preprocessed_incidents' \
  'gs://your-backup-bucket/incidents-$(date +%Y%m%d).parquet'
```

### **2. Model Backup**
```python
# Automated model backup to secondary storage
def backup_models():
    # Copy from primary blob storage to backup storage
    source_client = BlobServiceClient.from_connection_string(PRIMARY_CONNECTION)
    backup_client = BlobServiceClient.from_connection_string(BACKUP_CONNECTION)
    # Implementation...
```

### **3. Disaster Recovery Plan**
```markdown
## Recovery Procedures

1. **Data Loss Recovery**:
   - Restore from latest BigQuery export
   - Reprocess missing data from source

2. **Model Loss Recovery**:
   - Restore models from backup blob storage
   - If needed, retrain models from preprocessed data

3. **Infrastructure Failure**:
   - Deploy to backup region
   - Update DNS/endpoints
   - Validate functionality
```

## ✅ **Post-Deployment Validation**

### **1. End-to-End Test**
```bash
# Full pipeline test
python main.py preprocess --limit 100
python main.py train --tech-centers "BT-TC-Data Analytics" --year 2025 --quarter q2
python main.py predict --tech-center "BT-TC-Data Analytics"

# Verify results
python main.py status
```

### **2. Performance Validation**
```bash
# Load testing
python -c "
import time
start = time.time()
# Run prediction on 1000 incidents
end = time.time()
print(f'Processing time: {end - start:.2f} seconds')
print(f'Throughput: {1000 / (end - start):.2f} incidents/second')
"
```

### **3. Cost Monitoring**
```bash
# Check BigQuery costs
bq query --use_legacy_sql=false \
'SELECT 
  job_type,
  SUM(total_bytes_processed) / POW(10, 12) AS TB_processed,
  COUNT(*) AS job_count
FROM `your-project.region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY job_type'
```

## 🎯 **Success Criteria**

Your deployment is successful when:

- ✅ **All pipelines running**: Preprocessing, training, prediction
- ✅ **Data flowing**: New incidents processed within 1 hour
- ✅ **Models deployed**: Latest models in blob storage
- ✅ **Predictions generated**: Real-time classification working
- ✅ **Monitoring active**: Logs, alerts, health checks operational
- ✅ **Performance targets met**: <2 second prediction latency
- ✅ **Cost within budget**: Storage optimization achieved

**Your production HDBSCAN pipeline is now live! 🚀**