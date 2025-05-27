# HDBSCAN Clustering Pipeline

A complete, modularized HDBSCAN clustering pipeline for IT incidents with Azure OpenAI integration, checkpointing, and stage-wise execution.

## Features

- **Modular Architecture**: Run individual stages or the complete pipeline
- **Hybrid Embeddings**: Combines entity, action, and semantic embeddings
- **Azure OpenAI Integration**: Intelligent cluster labeling and domain grouping
- **Checkpointing**: Resume from any stage with saved intermediate results
- **BigQuery Integration**: Direct data loading and result saving
- **Error Handling**: Robust fallback mechanisms and logging
- **Scalable**: Handles large datasets with chunked processing

## Project Structure

```
hdbscan/
├── config/
│   ├── config.yaml          # Main configuration file
│   └── config.py             # Configuration loading utilities
├── data/
│   ├── bigquery_client.py    # BigQuery data operations
│   └── blob_storage.py       # Azure Blob Storage client
├── preprocessing/
│   ├── text_processing.py    # Text summarization and processing
│   └── embedding_generation.py # Hybrid embedding generation
├── core/
│   └── clustering.py         # HDBSCAN clustering with UMAP
├── analysis/
│   ├── cluster_analysis.py   # Cluster information generation
│   ├── cluster_labeling.py   # LLM-based cluster labeling
│   └── domain_grouping.py    # Hierarchical domain grouping
├── utils/
│   └── file_utils.py         # File operation utilities
├── main_pipeline.py          # Main pipeline orchestrator
├── main.py                   # Command-line interface
└── requirements.txt          # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Settings

Copy and edit the configuration file:

```yaml
# config/config.yaml
azure:
  openai_endpoint: "https://your-endpoint.openai.azure.com/"
  openai_api_key: "your-api-key"
  openai_embedding_endpoint: "https://your-embedding-endpoint.openai.azure.com/"
  openai_embedding_key: "your-embedding-key"
  api_version: "2024-02-15-preview"
  chat_model: "gpt-4o"
  embedding_model: "text-embedding-3-large"

bigquery:
  project_id: "your-project-id"
  credentials_path: "path/to/service-account.json"

pipeline:
  result_path: "./results"

clustering:
  min_cluster_size: 50
  min_samples: 25
  umap_n_components: 15
  umap_n_neighbors: 15
  umap_min_dist: 0.1
  max_domains: 20

embedding:
  batch_size: 25
  entity_weight: 0.3
  action_weight: 0.3
  semantic_weight: 0.4
```

### 3. Run the Pipeline

```bash
# Complete pipeline
python main.py --dataset incidents_2024 --query "SELECT * FROM your.incidents_table LIMIT 1000"

# Stage-wise execution
python main.py --dataset incidents_2024 --start-stage 1 --end-stage 2 --query "SELECT * FROM your.incidents_table"
python main.py --dataset incidents_2024 --start-stage 3 --end-stage 4
```

## Pipeline Stages

### Stage 1: Embedding Generation
- Loads data from BigQuery
- Processes text with Azure OpenAI for summarization
- Generates hybrid embeddings (entity + action + semantic)
- Saves embeddings locally and optionally to BigQuery

### Stage 2: HDBSCAN Clustering
- Applies UMAP for dimensionality reduction
- Performs HDBSCAN clustering
- Supports checkpointing for large datasets
- Saves cluster assignments and models

### Stage 3: Cluster Analysis
- Generates cluster information and samples
- Uses Azure OpenAI for intelligent cluster labeling
- Groups clusters into domains using hierarchical clustering
- Applies standardized labels across domains

### Stage 4: Results Export
- Saves final results to BigQuery and local files
- Handles large datasets with chunked processing
- Maintains data integrity and error logging

## Command-Line Options

```bash
# Required
--dataset DATASET_NAME    # Dataset name for organizing outputs

# Pipeline control
--start-stage {1,2,3,4}   # Stage to start from (default: 1)
--end-stage {1,2,3,4}     # Stage to end at (default: 4)

# Data input
--query "SQL_QUERY"       # BigQuery SQL query (required for stage 1)
--embedding-path PATH     # Path to existing embeddings (for stages 2+)
--summary-path PATH       # Path to existing summaries (for stage 1)

# BigQuery outputs
--embeddings-table TABLE  # BigQuery table for embeddings
--results-table TABLE     # BigQuery table for final results
--write-disposition MODE  # WRITE_APPEND or WRITE_TRUNCATE

# Configuration
--config CONFIG_PATH      # Path to config file (default: config/config.yaml)
--no-checkpoint          # Disable checkpointing
--log-level LEVEL        # DEBUG, INFO, WARNING, ERROR
```

## Configuration Options

### Clustering Parameters
- `min_cluster_size`: Minimum size for HDBSCAN clusters
- `min_samples`: Minimum samples for HDBSCAN core points
- `umap_n_components`: UMAP output dimensions
- `max_domains`: Maximum number of domains for grouping

### Embedding Parameters
- `batch_size`: Batch size for embedding generation
- `entity_weight`: Weight for entity embeddings
- `action_weight`: Weight for action embeddings  
- `semantic_weight`: Weight for semantic embeddings

## Output Structure

```
results/
└── {dataset_name}/
    ├── embeddings/
    │   ├── raw_data.parquet
    │   ├── df_with_embeddings.parquet
    │   └── embedding_metadata.json
    ├── intermediate/
    │   └── df_with_summaries.parquet
    ├── clustering/
    │   ├── clustered_df.parquet
    │   ├── umap_embeddings.npy
    │   ├── hdbscan_clusterer.pkl
    │   └── clustering_metadata.json
    ├── analysis/
    │   ├── final_df.parquet
    │   ├── cluster_details.json
    │   ├── labeled_clusters.json
    │   ├── domains.json
    │   └── analysis_metadata.json
    ├── final/
    │   └── results.csv
    └── run_metadata.json
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Azure OpenAI**: Retry logic with exponential backoff
- **Rate Limiting**: Automatic backoff and retry for API limits
- **Memory Management**: Chunked processing for large datasets
- **Checkpointing**: Resume from any stage without data loss
- **Fallback Labels**: Auto-generated labels if LLM fails
- **Data Validation**: Input validation and type checking

## Best Practices

1. **Start Small**: Test with a subset of data first
2. **Use Checkpointing**: Enable for large datasets to save time
3. **Monitor Logs**: Check logs for Azure OpenAI usage and errors
4. **Stage-wise Execution**: Run stages separately for debugging
5. **Resource Management**: Consider Azure OpenAI rate limits and costs

## Troubleshooting

### Common Issues

1. **Azure OpenAI Rate Limits**
   - Reduce `batch_size` in config
   - Check quota limits in Azure portal
   - Use retry logic (built-in)

2. **Memory Issues**
   - Reduce dataset size
   - Lower `umap_n_components`
   - Use checkpointing for large datasets

3. **BigQuery Errors**
   - Verify credentials and project ID
   - Check table permissions
   - Validate SQL query syntax

4. **Clustering Quality**
   - Adjust `min_cluster_size` and `min_samples`
   - Modify embedding weights
   - Check data quality and preprocessing

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for specific error messages
3. Create an issue with detailed information