# Preprocessing Pipeline Configuration Guide

## Overview

The preprocessing pipeline has been updated to handle your specific BigQuery data source and processing requirements:

**Data Source**: `enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev`

**Input Columns**: 
- `number` - Incident identifier
- `sys_created_on` - Creation timestamp  
- `description` - Detailed incident description
- `short_description` - Brief incident summary
- `business_service` - Affected application/service

**Processing Steps**:
1. **Text Cleaning**: Remove special characters, emails, URLs from all three text columns
2. **Column Combination**: Combine `description + short_description + business_service` with separator
3. **LLM Summarization**: Generate 30-word summary in format "what issues, which application got affected"
4. **Output**: New column `combined_incidents_summary` ready for embedding generation

## Configuration Changes Made

### 1. Updated `config.yaml`

```yaml
bigquery:
  tables:
    raw_incidents: "enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev"
  
  queries:
    incident_data_for_preprocessing: |
      SELECT 
        number,
        sys_created_on,
        description,
        short_description,
        business_service
      FROM `{source_table}`
      WHERE sys_created_on >= '{start_date}' 
      AND sys_created_on <= '{end_date}'
      ORDER BY sys_created_on DESC

preprocessing:
  text_columns_to_process: ["description", "short_description", "business_service"]
  text_column_for_summary_input: "combined_text_for_summary"
  summary_column_name: "combined_incidents_summary"
  
  clean_text:
    remove_special_characters: true
    remove_emails: true
    remove_urls: true
    normalize_whitespace: true
    min_text_length: 10
  
  summarization:
    enabled: true
    max_summary_length: 30  # Maximum words in summary
    summary_prompt_template: |
      Summarize the following incident information in exactly 30 words or less.
      Focus on: what issues occurred and which application/system was affected.
      Format: [Issue description] affecting [Application/System name].
      
      Incident Details:
      {combined_text}
      
      30-word Summary:
    model_name: "gpt-35-turbo"
    max_retries: 3
    batch_size: 5
```

### 2. Enhanced `TextProcessor` (`preprocessing/text_processing.py`)

**New Methods Added**:
- `process_incident_for_embedding_batch()` - Main method for processing incident DataFrames
- `_combine_incident_columns()` - Combines and cleans the three text columns
- `_build_summarization_prompt()` - Creates specific prompt for 30-word summaries

**Text Cleaning Features**:
- Removes special characters (keeps basic punctuation)
- Removes email addresses using regex
- Removes URLs  
- Normalizes whitespace
- Handles empty/missing values gracefully

### 3. Updated Pipeline Flow

```
BigQuery Data → Text Cleaning → Column Combination → LLM Summarization → Embedding Generation
```

**Before**: Expected tech_center grouping, hardcoded table names
**After**: Handles any incident data structure, configurable data source

## Usage Examples

### 1. Test with Sample Data

```bash
python examples/test_preprocessing_pipeline.py
```

This will:
- Create sample incidents matching your data structure
- Process them through the full pipeline
- Show the generated 30-word summaries
- Validate the output format

### 2. Production Usage

```python
from config.config import get_config
from pipeline.preprocessing_pipeline import PreprocessingPipeline

# Load configuration
config = get_config()

# Initialize pipeline  
preprocessing_pipeline = PreprocessingPipeline(config)

# Process your BigQuery data
# (The pipeline will automatically query your configured table)
results = await preprocessing_pipeline.process_for_training(dataset)
```

### 3. Manual BigQuery Integration

```python
from data_access.bigquery_client import BigQueryClient

# Query your specific table
query = """
SELECT number, sys_created_on, description, short_description, business_service
FROM `enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev`
WHERE sys_created_on >= '2024-01-01'
ORDER BY sys_created_on DESC
LIMIT 1000
"""

bigquery_client = BigQueryClient(config)
df = await bigquery_client.query_to_dataframe(query)

# Process the data
results = await preprocessing_pipeline.process_for_training(df)
```

## Expected Output Format

**Input Example**:
```
description: "SharePoint site not loading properly. Users unable to access documents."
short_description: "SharePoint access issue"  
business_service: "SharePoint Online"
```

**Combined Text**:
```
"SharePoint site not loading properly. Users unable to access documents. | SharePoint access issue | SharePoint Online"
```

**Generated Summary (30 words)**:
```
"SharePoint site loading failures preventing document access affecting SharePoint Online service for multiple users requiring immediate investigation and resolution."
```

## Validation & Testing

The pipeline includes comprehensive validation:

1. **Configuration Validation**: Ensures all required settings are present
2. **Data Quality Checks**: Validates input data structure and content
3. **Processing Statistics**: Tracks success rates and failure reasons
4. **Output Validation**: Verifies summary length and format

**Key Metrics Tracked**:
- Text cleaning success rate
- Summarization success rate  
- Word count compliance
- Processing duration
- Failed incident tracking

## Integration with Existing Pipeline

The preprocessing pipeline integrates seamlessly with the existing HDBSCAN training pipeline:

1. **Data Fetching**: Uses configured BigQuery query
2. **Preprocessing**: Generates `combined_incidents_summary` column
3. **Embedding Generation**: Processes summaries into vectors
4. **Training**: Feeds embeddings to HDBSCAN clustering
5. **Storage**: Saves results to versioned BigQuery tables

## Troubleshooting

**Common Issues**:

1. **"No tech_center column found"** - This is expected and handled automatically
2. **Azure OpenAI rate limits** - Pipeline includes retry logic and backoff
3. **Empty summaries** - Check Azure OpenAI configuration and API limits
4. **Data access errors** - Verify BigQuery permissions and table name

**Debug Steps**:
1. Run test script to validate configuration
2. Check logs for specific error messages
3. Verify Azure OpenAI endpoint and API key
4. Test BigQuery connection independently

This updated preprocessing pipeline now fully supports your specific data source and processing requirements! 🚀