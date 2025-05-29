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
    
    print("5. DEBUG LEVEL DETAILS (File Only):")
    print(DEBUG_LEVEL_LOGS)

if __name__ == "__main__":
    print_log_examples()