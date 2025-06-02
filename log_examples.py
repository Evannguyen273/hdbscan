# filepath: c:\Users\Bachn\OneDrive\Desktop\my_projects\hdbscan\log_examples.py
"""
Examples of what the detailed logging looks like during summarization pipeline execution.

This file shows actual log output examples for different scenarios.
"""

# Example 1: Successful Processing with Mixed Results
SUCCESSFUL_PROCESSING_LOGS = """
2024-01-15 14:30:15 | INFO     | text_processing | Starting summarization pipeline: 1000 incidents, batch_size=20
2024-01-15 14:30:15 | INFO     | text_processing | Processing batch 1/50 - incidents 1 to 20
2024-01-15 14:30:16 | DEBUG    | text_processing | Processing INC0012345: 1,234 chars, ~308 tokens
2024-01-15 14:30:17 | DEBUG    | text_processing | ✓ INC0012345: Successfully summarized
2024-01-15 14:30:17 | DEBUG    | text_processing | Processing INC0012346: 2,456 chars, ~614 tokens
2024-01-15 14:30:18 | WARNING  | text_processing | ✗ INC0012346: RATE_LIMIT - Rate limit exceeded. Please try again later.
2024-01-15 14:30:18 | DEBUG    | text_processing | Processing INC0012347: 345,678 chars, ~86,420 tokens
2024-01-15 14:30:18 | INFO     | text_processing | Text too long (86420 tokens), using chunking strategy for incident INC0012347
2024-01-15 14:30:22 | DEBUG    | text_processing | ✓ INC0012347: Successfully summarized
2024-01-15 14:30:35 | INFO     | text_processing | Batch 1 complete: 18/20 successful (90.0%) in 20.3s
2024-01-15 14:30:35 | WARNING  | text_processing | Batch 1 failures: ['INC0012346']
2024-01-15 14:30:36 | INFO     | text_processing | Processing batch 2/50 - incidents 21 to 40
...
2024-01-15 14:47:22 | INFO     | text_processing | Summarization complete: 950/1000 successful (95.0%)
2024-01-15 14:47:22 | WARNING  | text_processing | Failed incidents: ['INC0012346', 'INC0012401', 'INC0012567']
"""

# Example 2: High Failure Rate Scenario
HIGH_FAILURE_LOGS = """
2024-01-15 15:00:00 | INFO     | text_processing | Starting summarization pipeline: 100 incidents, batch_size=10
2024-01-15 15:00:00 | INFO     | text_processing | Processing batch 1/10 - incidents 1 to 10
2024-01-15 15:00:01 | WARNING  | text_processing | ✗ INC0013001: RATE_LIMIT - Rate limit exceeded. Please try again later.
2024-01-15 15:00:01 | WARNING  | text_processing | ✗ INC0013002: RATE_LIMIT - Rate limit exceeded. Please try again later.
2024-01-15 15:00:01 | WARNING  | text_processing | ✗ INC0013003: TIMEOUT - Request timed out after 30 seconds
2024-01-15 15:00:02 | WARNING  | text_processing | ✗ INC0013004: SERVICE_UNAVAILABLE - Service temporarily unavailable
2024-01-15 15:00:02 | DEBUG    | text_processing | ✓ INC0013005: Successfully summarized
2024-01-15 15:00:03 | WARNING  | text_processing | ✗ INC0013006: TOKEN_LIMIT - This model's maximum context length is 128000 tokens
2024-01-15 15:00:04 | INFO     | text_processing | Batch 1 complete: 1/10 successful (10.0%) in 4.2s
2024-01-15 15:00:04 | WARNING  | text_processing | Batch 1 failures: ['INC0013001', 'INC0013002', 'INC0013003', 'INC0013004', 'INC0013006']
"""

# Example 3: Token Limit and Chunking Scenario
CHUNKING_SCENARIO_LOGS = """
2024-01-15 16:15:30 | INFO     | text_processing | Processing batch 5/20 - incidents 81 to 100
2024-01-15 16:15:31 | DEBUG    | text_processing | Processing INC0014567: 456,789 chars, ~114,197 tokens
2024-01-15 16:15:31 | INFO     | text_processing | Text too long (114197 tokens), using chunking strategy for incident INC0014567
2024-01-15 16:15:32 | DEBUG    | text_processing | Summarizing chunk 1/3 for INC0014567
2024-01-15 16:15:34 | DEBUG    | text_processing | Summarizing chunk 2/3 for INC0014567
2024-01-15 16:15:36 | DEBUG    | text_processing | Summarizing chunk 3/3 for INC0014567
2024-01-15 16:15:36 | DEBUG    | text_processing | ✓ INC0014567: Successfully summarized (3 chunks combined)
2024-01-15 16:15:37 | DEBUG    | text_processing | Processing INC0014568: 890,123 chars, ~222,531 tokens
2024-01-15 16:15:37 | WARNING  | text_processing | ✗ INC0014568: TOKEN_LIMIT - Text too large even for chunking strategy
"""

# Example 4: Final Summary Report
FINAL_SUMMARY_LOGS = """
2024-01-15 17:30:45 | INFO     | text_processing | Summarization complete: 4,756/5,000 successful (95.1%)
2024-01-15 17:30:45 | WARNING  | text_processing | Failed incidents: ['INC0012346', 'INC0012401', 'INC0012567', ...]
2024-01-15 17:30:45 | INFO     | text_processing | Failure breakdown:
2024-01-15 17:30:45 | INFO     | text_processing |   - RATE_LIMIT: 156 incidents
2024-01-15 17:30:45 | INFO     | text_processing |   - TIMEOUT: 45 incidents  
2024-01-15 17:30:45 | INFO     | text_processing |   - TOKEN_LIMIT: 23 incidents
2024-01-15 17:30:45 | INFO     | text_processing |   - API_ERROR: 20 incidents
2024-01-15 17:30:45 | INFO     | text_processing | Processing duration: 47.2 minutes
2024-01-15 17:30:45 | INFO     | text_processing | Average processing rate: 105.9 incidents/minute
"""

# Example 5: Debug Level Logs (File Only)
DEBUG_LEVEL_LOGS = """
2024-01-15 14:30:16 | DEBUG    | text_processing | Raw text lengths - short_desc: 45, desc: 1,189, business_svc: 23
2024-01-15 14:30:16 | DEBUG    | text_processing | Cleaned text lengths - short_desc: 44, desc: 1,156, business_svc: 22
2024-01-15 14:30:16 | DEBUG    | text_processing | Estimated tokens: input=295, output=100, total=395 (within limits)
2024-01-15 14:30:17 | DEBUG    | text_processing | API call successful in 0.85s
2024-01-15 14:30:17 | DEBUG    | text_processing | Generated summary: "SharePoint Online experiencing slow response times..."
"""

# Example 6: Cumulative Training with Versioned Tables
CUMULATIVE_TRAINING_LOGS = """
2024-01-15 18:00:00 | INFO     | training_pipeline | Starting cumulative training for BT-TC-Data Analytics - Q2 2025
2024-01-15 18:00:00 | INFO     | training_pipeline | Training window: July 2023 → June 2025 (24 months cumulative)
2024-01-15 18:00:01 | INFO     | training_pipeline | Loading incidents from preprocessed_incidents table
2024-01-15 18:00:05 | INFO     | training_pipeline | Loaded 87,456 incidents for cumulative training
2024-01-15 18:00:05 | INFO     | training_pipeline | Performing HDBSCAN clustering on 24-month dataset
2024-01-15 18:00:45 | INFO     | training_pipeline | Clustering complete: 23 clusters identified
2024-01-15 18:00:45 | INFO     | training_pipeline | Grouping clusters into domains with auto-optimization (max: 20 domains)
2024-01-15 18:01:15 | INFO     | training_pipeline | Selected optimal domain count: 18 (score: 0.847)
2024-01-15 18:01:15 | INFO     | training_pipeline | Processing domain 1/18 with 3 clusters (attempt 1)
2024-01-15 18:01:18 | INFO     | training_pipeline | Processing domain 2/18 with 2 clusters (attempt 1)
...
2024-01-15 18:05:22 | INFO     | training_pipeline | Domain grouping completed: 18 domains created
2024-01-15 18:05:22 | INFO     | training_pipeline | Creating versioned BigQuery table: clustering_predictions_2025_q2_789
2024-01-15 18:05:23 | INFO     | training_pipeline | Prepared 87,456 records for table clustering_predictions_2025_q2_789
2024-01-15 18:05:24 | INFO     | training_pipeline | Saved BigQuery results to results/BT-TC-Data Analytics_2025_q2/clustering_predictions_2025_q2_789.json
2024-01-15 18:05:24 | INFO     | training_pipeline | Cumulative training completed for BT-TC-Data Analytics
"""

# Example 7: Version Comparison Logs
VERSION_COMPARISON_LOGS = """
2024-01-15 19:00:00 | INFO     | model_comparison | Comparing model versions for BT-TC-Data Analytics
2024-01-15 19:00:00 | INFO     | model_comparison | Previous model: clustering_predictions_2024_q4_789 (18 domains, 65,234 records)
2024-01-15 19:00:00 | INFO     | model_comparison | Current model: clustering_predictions_2025_q2_789 (18 domains, 87,456 records)
2024-01-15 19:00:01 | INFO     | model_comparison | Domain evolution analysis:
2024-01-15 19:00:01 | INFO     | model_comparison |   - 3 domains merged due to similar patterns
2024-01-15 19:00:01 | INFO     | model_comparison |   - 2 new domains emerged for Q1 2025 patterns
2024-01-15 19:00:01 | INFO     | model_comparison |   - 1 domain split due to pattern divergence
2024-01-15 19:00:01 | INFO     | model_comparison | Model ready for deployment
"""

# Example 8: Blob Storage Model Upload/Download Logs
BLOB_STORAGE_LOGS = """
2024-01-15 18:05:25 | INFO     | clustering_trainer | Uploading model artifacts to blob storage
2024-01-15 18:05:25 | INFO     | clustering_trainer | Blob container: hdbscan-models
2024-01-15 18:05:25 | INFO     | clustering_trainer | Blob prefix: bt-tc-data-analytics/2025_q2/
2024-01-15 18:05:26 | INFO     | clustering_trainer | Uploading umap_model.pkl (2.4 MB)
2024-01-15 18:05:28 | INFO     | clustering_trainer | Uploading hdbscan_model.pkl (1.8 MB)
2024-01-15 18:05:30 | INFO     | clustering_trainer | Uploading umap_embeddings.npy (156.7 MB)
2024-01-15 18:05:45 | INFO     | clustering_trainer | Uploading cluster_labels.npy (0.7 MB)
2024-01-15 18:05:46 | INFO     | clustering_trainer | Uploading model_metadata.json (2.1 KB)
2024-01-15 18:05:46 | INFO     | clustering_trainer | Model artifacts uploaded successfully to blob storage
2024-01-15 18:05:46 | INFO     | clustering_trainer | Total blob storage: 161.6 MB for model version 2025_q2
"""

# Example 9: Prediction Pipeline with Blob Storage
PREDICTION_BLOB_LOGS = """
2024-01-15 20:15:00 | INFO     | prediction_pipeline | Starting predictions for BT-TC-Data Analytics using model 2025_q2
2024-01-15 20:15:00 | INFO     | prediction_pipeline | Using trained model from table: clustering_predictions_2025_q2_789
2024-01-15 20:15:01 | INFO     | prediction_pipeline | Loading model artifacts from blob storage: hdbscan-models/bt-tc-data-analytics/2025_q2/
2024-01-15 20:15:02 | INFO     | prediction_pipeline | Downloaded umap_model.pkl from blob storage (2.4 MB)
2024-01-15 20:15:03 | INFO     | prediction_pipeline | Downloaded hdbscan_model.pkl from blob storage (1.8 MB)
2024-01-15 20:15:04 | INFO     | prediction_pipeline | Downloaded umap_embeddings.npy from blob storage (156.7 MB)
2024-01-15 20:15:05 | INFO     | prediction_pipeline | Model artifacts loaded from blob storage: bt-tc-data-analytics/2025_q2/
2024-01-15 20:15:05 | INFO     | prediction_pipeline | Loading domain mappings from BigQuery table: clustering_predictions_2025_q2_789
2024-01-15 20:15:06 | INFO     | prediction_pipeline | Domain mappings loaded from clustering_predictions_2025_q2_789: 23 clusters
2024-01-15 20:15:06 | INFO     | prediction_pipeline | Loaded model: 23 clusters, 18 domains
2024-01-15 20:15:06 | INFO     | prediction_pipeline | Model source: Blob storage + BigQuery table clustering_predictions_2025_q2_789
2024-01-15 20:15:07 | INFO     | prediction_pipeline | Generating predictions for 150 new incidents
2024-01-15 20:15:10 | INFO     | prediction_pipeline | Predictions generated successfully
2024-01-15 20:15:10 | INFO     | prediction_pipeline | Saving 150 predictions to BigQuery for BT-TC-Data Analytics
2024-01-15 20:15:11 | INFO     | prediction_pipeline | Model table referenced: clustering_predictions_2025_q2_789
2024-01-15 20:15:11 | INFO     | prediction_pipeline | Predictions saved to incident_predictions table (without embeddings for cost optimization)
2024-01-15 20:15:11 | INFO     | prediction_pipeline | Completed predictions for BT-TC-Data Analytics: 150 incidents
"""

def print_log_examples():
    """Print examples of log output for different scenarios"""
    print("=== LOG EXAMPLES FOR SUMMARIZATION PIPELINE ===\n")
    
    print("1. SUCCESSFUL PROCESSING WITH MIXED RESULTS:")
    print(SUCCESSFUL_PROCESSING_LOGS)
    print("\n" + "="*60 + "\n")
    
    print("2. HIGH FAILURE RATE SCENARIO:")
    print(HIGH_FAILURE_LOGS)
    print("\n" + "="*60 + "\n")
    
    print("3. CHUNKING FOR LARGE TEXTS:")
    print(CHUNKING_SCENARIO_LOGS)
    print("\n" + "="*60 + "\n")
    
    print("4. FINAL SUMMARY REPORT:")
    print(FINAL_SUMMARY_LOGS)
    print("\n" + "="*60 + "\n")
    
    print("5. DEBUG LEVEL LOGS:")
    print(DEBUG_LEVEL_LOGS)
    print("\n" + "="*60 + "\n")
    
    print("6. CUMULATIVE TRAINING WITH VERSIONED TABLES:")
    print(CUMULATIVE_TRAINING_LOGS)
    print("\n" + "="*60 + "\n")
    
    print("7. VERSION COMPARISON LOGS:")
    print(VERSION_COMPARISON_LOGS)
    print("\n" + "="*60 + "\n")
    
    print("8. BLOB STORAGE MODEL UPLOAD/DOWNLOAD LOGS:")
    print(BLOB_STORAGE_LOGS)
    print("\n" + "="*60 + "\n")
    
    print("9. PREDICTION PIPELINE WITH BLOB STORAGE:")
    print(PREDICTION_BLOB_LOGS)

if __name__ == "__main__":
    print_log_examples()