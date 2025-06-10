# %%
# Standard library imports
import json
import logging
import os
import pickle
import re
import time
import uuid
import warnings
from datetime import datetime

# Third-party imports - Data processing and analysis
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Azure services
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError, ServiceRequestError
from azure.identity import DefaultAzureCredential
import logging

# AI and ML
import hdbscan
import openai
import spacy
import umap
# from openai import AzureOpenAI
from openai import OpenAI
# from langchain.llms import AzureOpenAI

# Big data
import pyarrow.parquet as pq
from google.cloud import bigquery

# Setup
warnings.filterwarnings('ignore')
load_dotenv(r"C:\Users\BANGU\OneDrive - H & M HENNES & MAURITZ GBC AB\Documents\GitHub\06. Latest API_dev\hm-iservice-agentstore\.credenv")  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Setup Azure OpenAI API

# Load environment variables - with fallback paths
load_dotenv(r"C:\Users\BANGU\OneDrive - H & M HENNES & MAURITZ GBC AB\Documents\GitHub\06. Latest API_dev\hm-iservice-agentstore\.credenv")
# load_dotenv(".env", override=False)  # Try a regular .env file as backup

# Define defaults if environment variables are missing
azure_openai_config = {
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
    "chat_deployment_name": os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o"),
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "embedding_key": os.environ.get("AZURE_OPENAI_EMBEDDING_KEY"),
    "embedding_api_version": os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    "embedding_endpoint": os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    "embedding_model": os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")
}

# Validate required configurations
missing_configs = [key for key, value in azure_openai_config.items()
                   if value is None and key in ["api_key", "endpoint", "embedding_key", "embedding_endpoint"]]

if missing_configs:
    raise ValueError(f"Missing required Azure OpenAI configurations: {', '.join(missing_configs)}")

# Initialize Azure OpenAI client for chat/completions
openai_client = openai.AzureOpenAI(
    azure_endpoint=azure_openai_config["endpoint"],
    api_key=azure_openai_config["api_key"],
    api_version=azure_openai_config["api_version"],
)

# %%
embedding_client = openai.AzureOpenAI(
    api_key=azure_openai_config["embedding_key"],
    api_version=azure_openai_config["embedding_api_version"],  # Make sure to use a supported API version
    azure_endpoint="https://oai-insights-prod-001.openai.azure.com/"  # Use the base endpoint URL
)

def get_embedding(text):
    response = embedding_client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# Example usage
text = "This is a sample text for embedding"
embedding = get_embedding(text)
print(f"Generated embedding with {len(embedding)} dimensions")


# %%
# BigQuery setup
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import bigquery_storage

def connect_to_bq():
    """Connect to BigQuery using credentials from environment"""
    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("service_account_key_path")
    if not GOOGLE_APPLICATION_CREDENTIALS:
        raise ValueError("Environment variable 'service_account_key_path' is not set.")

    try:
        # Check if the credentials are a file path or JSON string
        if os.path.isfile(GOOGLE_APPLICATION_CREDENTIALS):
            credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
        else:
            json_key = json.loads(GOOGLE_APPLICATION_CREDENTIALS)
            credentials = service_account.Credentials.from_service_account_info(json_key)

        client = bigquery.Client(credentials=credentials)
        storage_client = bigquery_storage.BigQueryReadClient(credentials=credentials)
        logging.info('Connected to BigQuery')
        return client, storage_client
    except Exception as e:
        raise ConnectionError(f"Failed to connect to BigQuery: {e}")

def run_query(query, max_retries=3):
    """Execute a query with retry logic and return results as DataFrame"""
    retry_count = 0
    backoff_time = 2  # Start with 2 seconds backoff

    while retry_count <= max_retries:
        try:
            client, _ = connect_to_bq()
            query_job = client.query(query)
            query_result = query_job.result()
            return query_result.to_dataframe(create_bqstorage_client=False)
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                logging.error(f"Query failed after {max_retries} attempts: {e}")
                return pd.DataFrame()

            logging.warning(f"Query attempt {retry_count} failed: {e}. Retrying in {backoff_time}s")
            import time
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff


def robust_json_parser(json_string):
    """
    Robust JSON parser optimized for Azure OpenAI responses with enhanced error recovery.
    Handles unterminated strings, delimiter issues, and malformed JSON with multiple fallback strategies.

    Args:
        json_string: JSON string that might be malformed

    Returns:
        Parsed JSON object or fallback dictionary
    """
    try:
        # First try standard parsing
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        position = e.pos
        error_type = str(e)
        logging.warning(f"JSON parsing error at position {position}: {error_type}")

        # Initialize fixed string
        fixed_string = json_string

        # PHASE 1: STRING TERMINATION & ESCAPING FIX
        if "Unterminated string" in error_type or "Invalid \escape" in error_type:
            logging.info(f"Fixing unterminated string or escape sequence")
            # Build character by character with proper state tracking
            in_string = False
            escape_next = False
            string_start_pos = None
            fixed_chars = []

            # Add proper tracking of brackets and braces for better context
            brackets_stack = []

            for i, c in enumerate(fixed_string):
                # Handle escape sequences
                if escape_next:
                    escape_next = False
                    fixed_chars.append(c)
                    continue

                # Handle string boundaries
                if c == '"' and not escape_next:
                    if not in_string:
                        in_string = True
                        string_start_pos = i
                    else:
                        in_string = False
                        string_start_pos = None
                # Track escape character but only in strings
                elif c == '\\' and in_string:
                    escape_next = True
                # Track structure characters when not in strings
                elif not in_string:
                    if c in '{[':
                        brackets_stack.append(c)
                    elif c in '}]':
                        if brackets_stack and ((c == '}' and brackets_stack[-1] == '{') or
                                              (c == ']' and brackets_stack[-1] == '[')):
                            brackets_stack.pop()

                fixed_chars.append(c)

            # Close any unclosed strings
            if in_string:
                logging.info(f"Closing unterminated string that started at position {string_start_pos}")
                fixed_chars.append('"')

            # Close any unclosed brackets/braces
            for bracket in reversed(brackets_stack):
                if bracket == '{':
                    fixed_chars.append('}')
                elif bracket == '[':
                    fixed_chars.append(']')

            fixed_string = ''.join(fixed_chars)

            try:
                return json.loads(fixed_string)
            except json.JSONDecodeError as e2:
                logging.warning(f"String termination fix failed: {e2}")
                # Continue to next repair strategy

        # PHASE 2: STRUCTURAL REPAIRS

        # Remove non-printable characters
        filtered_string = ''.join(c for c in fixed_string if c.isprintable() or c.isspace())
        if filtered_string != fixed_string:
            logging.info("Removed non-printable characters")
            fixed_string = filtered_string

            try:
                return json.loads(fixed_string)
            except json.JSONDecodeError:
                pass

        # Fix missing quotes around keys
        fixed_string = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', fixed_string)

        # Fix missing commas between objects and array elements
        fixed_string = re.sub(r'}\s*{', '},{', fixed_string)
        fixed_string = re.sub(r'"\s*{', '",{', fixed_string)
        fixed_string = re.sub(r'}\s*"', '},"', fixed_string)
        fixed_string = re.sub(r']\s*\[', '],[', fixed_string)
        fixed_string = re.sub(r']\s*{', '],{', fixed_string)
        fixed_string = re.sub(r'}\s*\[', '},[', fixed_string)

        # Fix issue with missing comma specifically at position mentioned in error
        if "Expecting ',' delimiter" in error_type and position > 0:
            # Try to fix the specific position mentioned in the error
            logging.info(f"Adding missing comma delimiter near position {position}")
            try:
                error_pos = position - 10  # Look a bit before the reported position
                error_pos = max(0, error_pos)
                search_segment = fixed_string[error_pos:position+10]

                # Common patterns where commas might be missing
                patterns = [
                    (r'"\s*"', '","'),           # "value""next" -> "value","next"
                    (r'"\s*{', '",{'),           # "value"{next -> "value",{next
                    (r'}\s*"', '},"'),           # }"value" -> },"value"
                    (r']\s*"', '],"'),           # ]"value" -> ],"value"
                    (r'"\s*\[', '",['),          # "value"[next -> "value",[next
                    (r'true\s*"', 'true,"'),     # true"value" -> true,"value"
                    (r'false\s*"', 'false,"'),   # false"value" -> false,"value"
                    (r'null\s*"', 'null,"'),     # null"value" -> null,"value"
                    (r'(\d+)\s*"', r'\1,"'),     # 123"value" -> 123,"value"
                    (r'"\s*(\d+)', '",\1')       # "value"123 -> "value",123
                ]

                for pattern, replacement in patterns:
                    search_segment_new = re.sub(pattern, replacement, search_segment)
                    if search_segment_new != search_segment:
                        # Replace just this segment in the full string
                        fixed_string = fixed_string[:error_pos] + search_segment_new + fixed_string[error_pos+len(search_segment):]
                        break
            except Exception as segment_error:
                logging.warning(f"Error during delimiter fix: {segment_error}")

        # Fix trailing/leading commas in arrays/objects
        fixed_string = re.sub(r',\s*}', '}', fixed_string)
        fixed_string = re.sub(r',\s*]', ']', fixed_string)

        try:
            return json.loads(fixed_string)
        except json.JSONDecodeError as e3:
            logging.warning(f"Structural repairs failed: {e3}")

        # PHASE 3: EXTRACTION STRATEGIES
        logging.info("Attempting extraction strategies")

        # Strategy 1: Extract the outermost JSON object
        try:
            start = fixed_string.find('{')
            end = fixed_string.rfind('}')
            if start != -1 and end != -1 and start < end:
                extracted = fixed_string[start:end+1]
                logging.info(f"Extracting outermost JSON object from position {start} to {end}")
                return json.loads(extracted)
        except Exception:
            pass

        # Strategy 2: Find and extract objects with topic/description pattern (common in cluster responses)
        try:
            # Match pattern for cluster info
            pattern = r'"([^"]+)":\s*{\s*"topic":\s*"([^"]*)"(?:[^}]*)"description":\s*"([^"]*)"'
            matches = re.findall(pattern, fixed_string)
            if matches:
                logging.info(f"Found {len(matches)} cluster objects using pattern extraction")
                result = {}
                for cluster_id, topic, description in matches:
                    result[cluster_id] = {
                        "topic": topic,
                        "description": description
                    }
                return result
        except Exception:
            pass

        # Strategy 3: Look for partial JSON structure recovery
        try:
            # Try to find valid JSON objects within the string
            brace_level = 0
            start_pos = None
            partial_jsons = []

            for i, c in enumerate(fixed_string):
                if c == '{':
                    if brace_level == 0:
                        start_pos = i
                    brace_level += 1
                elif c == '}':
                    brace_level -= 1
                    if brace_level == 0 and start_pos is not None:
                        # Found a complete JSON object
                        try:
                            obj = json.loads(fixed_string[start_pos:i+1])
                            partial_jsons.append(obj)
                        except:
                            pass

            if partial_jsons:
                logging.info(f"Found {len(partial_jsons)} valid JSON objects within malformed JSON")
                # If we have multiple objects, combine them
                if len(partial_jsons) > 1:
                    combined = {}
                    for obj in partial_jsons:
                        combined.update(obj)
                    return combined
                else:
                    return partial_jsons[0]
        except Exception:
            pass

        # PHASE 4: LAST RESORT STRATEGIES
        logging.warning("All JSON repair strategies failed, attempting last resort methods")

        # Try json5 parser (more permissive) if available
        try:
            import json5
            logging.info("Using json5 parser for lenient parsing")
            return json5.loads(fixed_string)
        except ImportError:
            logging.info("json5 module not available")
        except Exception:
            pass

        # Create minimal return object with any data we can extract
        try:
            # Regex for key-value patterns
            kv_pattern = r'"([^"]+)":\s*"([^"]+)"'
            kv_matches = re.findall(kv_pattern, fixed_string)

            if kv_matches:
                logging.info(f"Extracted {len(kv_matches)} key-value pairs as fallback")
                result = {k: v for k, v in kv_matches}
                return result
            else:
                logging.error("Could not extract any data from malformed JSON")
                return {}
        except Exception as final_error:
            logging.error(f"All JSON recovery methods failed: {final_error}")
            return {}

def validate_json_response(json_data, required_keys=None, schema=None):
    """
    Validate that a JSON response matches expected structure

    Args:
        json_data: Parsed JSON data to validate
        required_keys: List of top-level keys that must exist
        schema: Optional JSON schema for detailed validation

    Returns:
        (bool, str): Tuple of (is_valid, error_message)
    """
    # Check for empty response
    if not json_data:
        return False, "Empty JSON response"

    # Check required keys
    if required_keys:
        missing_keys = [key for key in required_keys if key not in json_data]
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"

    # Advanced schema validation if jsonschema module is available
    if schema:
        try:
            import jsonschema
            jsonschema.validate(instance=json_data, schema=schema)
        except ImportError:
            logging.warning("jsonschema module not available, skipping schema validation")
        except jsonschema.exceptions.ValidationError as e:
            return False, f"Schema validation error: {e.message}"

    return True, "Valid JSON"

def save_results_to_bigquery(df, table_id, write_disposition="WRITE_APPEND"):
    """Save final results to BigQuery"""

    logging.info(f"Saving {len(df)} results to {table_id}")

    # Select only the required columns
    required_columns = [
        'number', 'priority', 'sys_created_on', 'TeamDepartment', 'Area',
        'Unit', 'TechCenter', 'short_description', 'close_code', 'vendor',
        'assignment_group', 'state', 'business_service', 'close_notes',
        'work_notes', 'chanel', 'description', 'cluster', 'subcategory',
        'category'
    ]

    # Filter columns that exist in the dataframe
    available_columns = [col for col in required_columns if col in df.columns]
    results_df = df[available_columns].copy()

    # Connect to BigQuery
    client, _ = connect_to_bq()
    sanitized_table_id = table_id.replace('`', '')

    # Configure job
    write_disp = getattr(bigquery.WriteDisposition, write_disposition)
    job_config = bigquery.LoadJobConfig(write_disposition=write_disp)

    # Execute load job
    try:
        job = client.load_table_from_dataframe(
            results_df,
            sanitized_table_id,
            job_config=job_config,
            timeout=300  # 5 minutes timeout
        )
        job.result()  # Wait for job to complete
        logging.info(f"Successfully saved {len(results_df)} results to {table_id}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise


# %%
# Custom JSON Encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

# Update all json.dump calls to use this encoder
# Example: json.dump(data, f, indent=2, cls=NumpyEncoder)


# %%
from azure.storage.blob import ContentSettings
# Update all these instances
content_settings = ContentSettings(content_type="application/octet-stream")
container_name = "prediction-artifact"
sas_url = os.environ.get("BLOB_CONNECTION_STRING")
container_client = ContainerClient.from_container_url(sas_url)

# %%
def classify_terms_with_llm(df, text_column='combined_incidents_summary'):
    """
    Use GPT to classify terms from incidents into entities and actions
    Returns dict with ENTITY and ACTION categories and their frequencies
    """
    # Combine all text for term extraction (limit to avoid token issues)
    sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
    all_text = " ".join(sample_df[text_column].fillna('').astype(str).tolist())

    # Define JSON schema for better response structure
    json_schema = {
        "type": "object",
        "properties": {
            "ENTITY": {
                "type": "object",
                "additionalProperties": {"type": "number"}
            },
            "ACTION": {
                "type": "object",
                "additionalProperties": {"type": "number"}
            }
        },
        "required": ["ENTITY", "ACTION"]
    }

    # Use GPT to extract and classify terms with improved JSON instructions
    prompt = f"""
    Analyze the following IT incidents text and identify:
    1. ENTITIES: Technical components, systems, software, or services mentioned (like SAP, Outlook, VPN)
    2. ACTIONS: Verbs describing what happened or needs to happen (like crashed, failed, reset)

    Extract up to 50 most common entities and up to 50 most common actions.

    YOU MUST RESPOND WITH VALID JSON using this structure:
    {{
        "ENTITY": {{"term1": frequency, "term2": frequency, ...}},
        "ACTION": {{"term1": frequency, "term2": frequency, ...}}
    }}

    Incident text sample:
    {all_text[:3000]}
    """

    # Azure OpenAI best practices implementation
    correlation_id = str(uuid.uuid4())
    retry_attempts = 3

    for attempt in range(retry_attempts):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert IT analyst that extracts and classifies terms from incident descriptions. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent JSON output
                response_format={"type": "json_object"},
                timeout=30,
                user=correlation_id  # Track the request in Azure logs
            )

            # Use the robust JSON parser to handle any response issues
            result = robust_json_parser(response.choices[0].message.content)

            # Validate expected structure
            if "ENTITY" in result and "ACTION" in result:
                return result
            else:
                raise ValueError("Response missing required ENTITY or ACTION keys")

        except Exception as e:
            if attempt < retry_attempts - 1:
                wait_time = min(30, 2 ** attempt * 2)  # Exponential backoff
                logging.warning(f"Term classification attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                logging.error(f"All term classification attempts failed: {e}")
                # Return empty results as fallback
                return {"ENTITY": {}, "ACTION": {}}

# %%
def save_embeddings_to_bigquery(df, table_id, write_disposition="WRITE_APPEND"):
    """Save embeddings to BigQuery for persistence"""

    logging.info(f"Saving {len(df)} embeddings to {table_id}")

    # Create a copy with number, combined_incidents_summary, and embedding columns
    embedding_df = df[['number', 'combined_incidents_summary', 'embedding']].copy()

    # Connect to BigQuery
    client, _ = connect_to_bq()
    sanitized_table_id = table_id.replace('`', '')

    # Configure job
    write_disp = getattr(bigquery.WriteDisposition, write_disposition)
    job_config = bigquery.LoadJobConfig(write_disposition=write_disp)

    # Execute load job with retries and timeout
    try:
        job = client.load_table_from_dataframe(
            embedding_df,
            sanitized_table_id,
            job_config=job_config,
            timeout=300  # 5 minutes timeout
        )
        job.result()  # Wait for job to complete
        logging.info(f"Successfully saved {len(embedding_df)} embeddings to {table_id}")
    except Exception as e:
        logging.error(f"Error saving embeddings: {e}")
        # Continue pipeline even if saving fails
        pass

# %%
def get_safe_text(row, column):
    """Helper function to safely get text from a DataFrame row column"""
    return str(row[column]) if pd.notna(row[column]) else ""

def estimate_tokens(text):
    """
    Estimate token count for a given text using a simple approximation.

    Args:
        text: Input text string

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # OpenAI models generally use ~4 chars per token for English text
    return len(text) // 4 + 1  # Add 1 as a safety margin



def clean_text_for_summary(text):
    """
    Clean text by normalizing whitespace and removing special characters

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\r\t]+', ' ', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c.isspace())

    return text.strip()

def normalize_business_service(text):
    """
    Normalize business service names by removing dashes, underscores and standardizing spacing

    Args:
        text: Business service name

    Returns:
        Normalized business service name
    """
    if not text:
        return ""

    # Remove trailing " - PROD", " - DEV", etc.
    text = re.sub(r'\s*[-_]\s*(PROD|DEV|TEST|UAT|QA)$', '', text)

    # Replace dashes and underscores with spaces
    text = re.sub(r'[-_]+', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def summarize_incident_with_llm(short_desc, desc, business_service):
    """
    Use Azure OpenAI to generate a concise summary of the incident

    Args:
        short_desc: Short description of the incident
        desc: Full description of the incident
        business_service: Business service affected

    Returns:
        Concise summary of the incident
    """
    # Construct the prompt for summarization with more specific instructions
    prompt = f"""Create a single concise sentence (max 30 words) summarizing this IT incident.
Focus on: 1) the specific platform/application affected and 2) the exact issue or error.

SHORT DESCRIPTION: {short_desc}
DESCRIPTION: {desc}
AFFECTED SYSTEM: {business_service}

Your summary MUST:
1. Begin by clearly identifying the affected platform/application
2. Describe the specific technical issue (not working, error, failure, etc.)
3. Capture any key error messages or symptoms mentioned
4. Be technical but clear and concise (one sentence only)

Example summaries:
- "Adobe Campaign platform prevents users from publishing and unpublishing content due to authorization errors"
- "SAP system fails to process financial transactions with timeout errors during month-end closing"
- "Microsoft Teams meetings crash when screen sharing is attempted from macOS devices"
"""

    # Call Azure OpenAI with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a technical IT incident summarizer that creates precise, concise summaries focusing on affected systems and specific issues."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100,
                timeout=15  # 15 second timeout
            )
            summary = response.choices[0].message.content.strip()
            # Remove any quotes that might be in the response
            summary = summary.strip('"\'')
            return summary

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.warning(f"Summarization attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                logging.error(f"All summarization attempts failed: {e}")
                # Fall back to simply returning the short description
                return short_desc or "No description available"

def process_incident_for_embedding_batch(df, batch_size=10):
    """
    Process incident text for embedding with LLM summarization in batches

    Args:
        df: DataFrame with incidents
        batch_size: Number of incidents to process in each batch

    Returns:
        Tuple of (result_series, fallback_stats): Series with summarized texts and dictionary of fallback statistics
    """
    import uuid
    from tqdm.auto import tqdm

    result_series = pd.Series(index=df.index)
    num_batches = (len(df) + batch_size - 1) // batch_size

    # Initialize fallback tracking statistics
    fallback_stats = {
        "total_incidents": len(df),
        "short_desc_fallbacks": 0,
        "api_failure_fallbacks": 0,
        "final_sweep_fallbacks": 0,
        "llm_processed": 0
    }

    # Initialize progress bar
    pbar = tqdm(total=len(df), desc="Summarizing incidents")

    # Process in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        # Prepare batch data for API call
        batch_items = []
        for idx, row in batch_df.iterrows():
            # Get text fields
            short_desc = get_safe_text(row, 'short_description')
            desc = get_safe_text(row, 'description')
            business_svc = get_safe_text(row, 'business_service')

            # Clean and normalize
            clean_short_desc = clean_text_for_summary(short_desc)
            clean_desc = clean_text_for_summary(desc)
            clean_desc = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', clean_desc)
            clean_business_svc = normalize_business_service(business_svc)

            # Use simpler approach for short descriptions
            if len(clean_desc) < 300:
                result_series[idx] = f"{clean_short_desc} - {clean_business_svc}".strip()
                fallback_stats["short_desc_fallbacks"] += 1
                pbar.update(1)
                continue

            # For longer texts, add to batch for summarization
            batch_items.append({
                "idx": idx,
                "short_desc": clean_short_desc,
                "description": clean_desc[:1500],  # Truncate long descriptions
                "business_service": clean_business_svc
            })

        # If batch is empty after filtering short descriptions, continue
        if not batch_items:
            continue

        # Count number of items going to LLM for summarization
        fallback_stats["llm_processed"] += len(batch_items)

        # Create prompt for batch summarization
        batch_request_id = str(uuid.uuid4())
        prompt = "Summarize each of these IT incidents in a single concise sentence (max 30 words).\n\n"

        for i, item in enumerate(batch_items):
            prompt += f"INCIDENT #{i+1}:\n"
            prompt += f"SHORT DESCRIPTION: {item['short_desc']}\n"
            prompt += f"DESCRIPTION: {item['description']}\n"
            prompt += f"AFFECTED SYSTEM: {item['business_service']}\n\n"

        prompt += """For each incident, your summary MUST:
1. Begin by clearly identifying the affected platform/application
2. Describe the specific technical issue
3. Be technical but clear and concise (one sentence only)

RESPOND IN JSON FORMAT:
{
  "summaries": [
    {"incident": 1, "summary": "Platform X fails with error Y when performing action Z"},
    {"incident": 2, "summary": "..."}
    // etc. for each incident
  ]
}
"""

        # Make batch API call with retry logic and proper Azure instrumentation
        max_retries = 3
        batch_failures = 0
        for attempt in range(max_retries):
            try:
                # Add JSON schema validation and guidance
                correlation_id = str(uuid.uuid4())
                json_schema = {
                    "type": "object",
                    "properties": {
                        "summaries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "incident": {"type": "number"},
                                    "summary": {"type": "string"}
                                },
                                "required": ["incident", "summary"]
                            }
                        }
                    },
                    "required": ["summaries"]
                }

                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a technical IT incident summarizer that creates precise, concise summaries. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Lower temperature for more consistent JSON output
                    response_format={"type": "json_object"},  # Enforce JSON format
                    timeout=60,
                    user=correlation_id  # For request tracking
                )

                # Parse response with robust parser
                response_text = response.choices[0].message.content
                result = robust_json_parser(response_text)

                # Validate structure
                if "summaries" not in result or not isinstance(result["summaries"], list):
                    raise ValueError("Response missing required 'summaries' array")

                # Map summaries back to original indices
                for i, summary_obj in enumerate(result.get("summaries", [])):
                    if i < len(batch_items) and "summary" in summary_obj:
                        idx = batch_items[i]["idx"]
                        summary = summary_obj.get("summary", "")
                        business_svc = batch_items[i]["business_service"]

                        # If business service isn't in summary, append it
                        if business_svc and business_svc.lower() not in summary.lower():
                            summary = f"{summary} - {business_svc}"

                        result_series[idx] = summary

                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = min(30, 2 ** attempt * 2)  # Exponential backoff with max 30s wait
                    logging.warning(f"Batch summarization attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All batch summarization attempts failed: {e}")

                    # Fall back to simple combination for failed items
                    for item in batch_items:
                        idx = item["idx"]
                        if idx not in result_series or pd.isna(result_series[idx]):
                            result_series[idx] = f"{item['short_desc']} - {item['business_service']}".strip()
                            batch_failures += 1

                    fallback_stats["api_failure_fallbacks"] += batch_failures

        # Update progress
        pbar.update(min(batch_size, len(batch_df)))

    pbar.close()

    # Final sweep - fill any missing values with their short description
    final_sweep_count = 0
    for idx in df.index:
        if idx not in result_series or pd.isna(result_series[idx]):
            short_desc = get_safe_text(df.loc[idx], 'short_description')
            business_svc = get_safe_text(df.loc[idx], 'business_service')
            business_svc = normalize_business_service(business_svc)
            result_series[idx] = f"{short_desc} - {business_svc}".strip()
            final_sweep_count += 1

    fallback_stats["final_sweep_fallbacks"] = final_sweep_count

    # Calculate success rate
    fallback_stats["total_fallbacks"] = (
        fallback_stats["short_desc_fallbacks"] +
        fallback_stats["api_failure_fallbacks"] +
        fallback_stats["final_sweep_fallbacks"]
    )
    fallback_stats["llm_success_count"] = fallback_stats["llm_processed"] - fallback_stats["api_failure_fallbacks"] - fallback_stats["final_sweep_fallbacks"]
    fallback_stats["llm_success_rate"] = fallback_stats["llm_success_count"] / max(1, fallback_stats["llm_processed"]) * 100
    fallback_stats["overall_llm_rate"] = fallback_stats["llm_success_count"] / len(df) * 100

    logging.info(f"Summarization fallback stats: {fallback_stats}")

    return result_series, fallback_stats

def create_hybrid_embeddings(df, batch_size=25, summary_path=None, result_path="results", dataset_name=None):
    """
    Create hybrid embeddings combining entity, action, and semantic information
    with optimized batch processing and intermediate file saving

    Args:
        df: DataFrame with incidents
        batch_size: Batch size for embedding API calls
        summary_path: Path to load precomputed summaries (skip summarization if provided)
        result_path: Base path for saving intermediate results
        dataset_name: Name of current dataset for organized file saving

    Returns:
        Tuple of (DataFrame with embeddings, classification results, fallback statistics)
    """
    logging.info(f"Creating hybrid embeddings for {len(df)} records")
    result_df = df.copy()

    # Define output directory for intermediate files if dataset_name is provided
    summary_output_dir = None
    if dataset_name:
        summary_output_dir = f"{result_path}/{dataset_name}/intermediate"
        os.makedirs(summary_output_dir, exist_ok=True)

    # STEP 1: Generate or load combined_incidents_summary
    if summary_path and os.path.exists(summary_path):
        # Load precomputed summaries
        logging.info(f"Loading precomputed summaries from {summary_path}")
        summary_df = pd.read_parquet(summary_path)
        result_df['combined_incidents_summary'] = summary_df['combined_incidents_summary']
        # We don't have fallback stats when loading precomputed summaries
        fallback_stats = {"precomputed": True, "loaded_from": summary_path}
    else:
        # Process text for embedding using batch processing
        logging.info("Processing text for embedding...")
        # Unpack the tuple return value (combined summaries and fallback statistics)
        combined_summaries, fallback_stats = process_incident_for_embedding_batch(
            result_df,
            batch_size=min(10, len(df))  # Use smaller batch for LLM summarization
        )
        result_df['combined_incidents_summary'] = combined_summaries

        # Save intermediate dataframe with combined_incidents_summary for future reuse
        if summary_output_dir:
            summary_save_path = f"{summary_output_dir}/df_with_summaries.parquet"
            logging.info(f"Saving intermediate dataframe with summaries to {summary_save_path}")
            # Save only essential columns
            summary_df = result_df[['number', 'combined_incidents_summary']].copy()
            summary_df.to_parquet(summary_save_path, index=False)

        # Log fallback statistics with detailed breakdown
        logging.info(f"Text summarization complete: {len(df)} total incidents processed")
        logging.info(f"LLM attempted summarization for {fallback_stats['llm_processed']} incidents")
        logging.info(f"Fallback stats: {fallback_stats['short_desc_fallbacks']} short descriptions used directly")
        logging.info(f"API failures: {fallback_stats['api_failure_fallbacks']}, final sweep fallbacks: {fallback_stats['final_sweep_fallbacks']}")
        logging.info(f"LLM success rate: {fallback_stats['llm_success_rate']:.2f}% of attempts, {fallback_stats['overall_llm_rate']:.2f}% of total incidents")

    # Rest of the function remains the same - process for entity/action classification and embeddings
    text_column = 'combined_incidents_summary'

    # First, get entity and action classification
    logging.info("Classifying terms into entities and actions...")
    classification_result = classify_terms_with_llm(result_df, text_column)

    # Extract entity and action terms
    entity_terms = list(classification_result.get("ENTITY", {}).keys())
    action_terms = list(classification_result.get("ACTION", {}).keys())
    logging.info(f"Using {len(entity_terms)} entities and {len(action_terms)} action terms")

    # Create vectorizers and generate entity/action embeddings
    vectorizers = {
        'entity': TfidfVectorizer(vocabulary=entity_terms, lowercase=True),
        'action': TfidfVectorizer(vocabulary=action_terms, lowercase=True)
    }

    # Generate and transform embeddings
    text_data = result_df[text_column].fillna('')
    matrices = {
        'entity': vectorizers['entity'].fit_transform(text_data),
        'action': vectorizers['action'].fit_transform(text_data)
    }

    # Convert sparse matrices to dense
    dense_matrices = {k: m.toarray() for k, m in matrices.items()}

    # Generate semantic embeddings with Azure OpenAI
    logging.info("Generating semantic embeddings...")
    semantic_embeddings = generate_semantic_embeddings(
        result_df[text_column],
        batch_size=batch_size
    )

    # Scale all embedding components
    logging.info("Scaling and combining embeddings...")
    scaled_matrices = {}
    for name, matrix in {**dense_matrices, 'semantic': semantic_embeddings}.items():
        scaler = StandardScaler()
        scaled_matrices[name] = scaler.fit_transform(matrix)

    # Set weights for each component
    weights = {'entity': 0.0, 'action': 0.0, 'semantic': 1.0}

    # Get dimensions for each component
    dims = {k: m.shape[1] for k, m in scaled_matrices.items()}

    # Create combined embeddings array
    total_dims = sum(dims.values())
    combined_embeddings = np.zeros((len(result_df), total_dims))

    # Fill combined embeddings with weighted components
    start_idx = 0
    for name, matrix in scaled_matrices.items():
        end_idx = start_idx + dims[name]
        combined_embeddings[:, start_idx:end_idx] = weights[name] * matrix
        start_idx = end_idx

    # Convert embeddings to JSON strings
    result_df['embedding'] = [json.dumps(emb.tolist()) for emb in combined_embeddings]

    # Save minimal dataframe with just number, summary and embedding if dataset_name provided
    if summary_output_dir:
        embedding_save_path = f"{summary_output_dir}/df_with_minimal_embeddings.parquet"
        logging.info(f"Saving minimal dataframe with embeddings to {embedding_save_path}")
        minimal_df = result_df[['number', 'combined_incidents_summary', 'embedding']].copy()
        minimal_df.to_parquet(embedding_save_path, index=False)

    logging.info("Hybrid embedding generation complete")
    return result_df, classification_result, fallback_stats



def generate_semantic_embeddings(text_series, batch_size=25, model="text-embedding-3-large"):
    """
    Generate semantic embeddings using Azure OpenAI with Azure best practices

    Args:
        text_series: Pandas Series containing text to embed
        batch_size: Size of batches for API calls
        model: Embedding model to use

    Returns:
        Numpy array of embeddings
    """
    import uuid

    semantic_embeddings = []
    total_batches = (len(text_series) + batch_size - 1) // batch_size
    retry_limit = 3

    # Azure-specific metrics tracking
    azure_metrics = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "rate_limited_requests": 0,
        "total_time": 0
    }

    # Reduce batch size for very large texts to avoid token limits
    max_estimated_tokens = text_series.apply(lambda x: estimate_tokens(str(x))).max()
    if max_estimated_tokens > 4000:
        adjusted_batch_size = max(1, min(batch_size, 5))
        logging.warning(f"Detected large texts (est. {max_estimated_tokens} tokens). Reducing batch size to {adjusted_batch_size}")
        batch_size = adjusted_batch_size
        total_batches = (len(text_series) + batch_size - 1) // batch_size

    # Create circuit breaker for resilience
    circuit_open = False
    circuit_failures = 0
    circuit_threshold = 5
    circuit_reset_time = None

    # Process in batches with improved progress tracking
    with tqdm(total=len(text_series), desc="Embedding texts") as pbar:
        for i in range(0, len(text_series), batch_size):
            # Check circuit breaker
            if circuit_open:
                if time.time() - circuit_reset_time > 60:  # 1 minute timeout
                    circuit_open = False
                    circuit_failures = 0
                    logging.info("Circuit breaker reset, resuming API calls")
                else:
                    logging.warning(f"Circuit breaker open, skipping batch and using zeros. Resets in {60 - (time.time() - circuit_reset_time):.0f}s")
                    semantic_embeddings.extend([[0.0] * 3072 for _ in range(min(batch_size, len(text_series) - i))])
                    pbar.update(min(batch_size, len(text_series) - i))
                    continue

            # Create batch with correlation ID for Azure request tracking
            batch = text_series.iloc[i:i+batch_size].fillna('').tolist()
            batch_size_actual = len(batch)
            correlation_id = str(uuid.uuid4())

            azure_metrics["total_requests"] += 1
            start_time = time.time()

            try:
                # Call the embedding API with Azure-specific headers
                # Modified version - removes headers and uses user parameter
                response = embedding_client.embeddings.create(
                    input=batch,
                    model=model,
                    timeout=30,
                    user=correlation_id  # Use the correlation_id as the user parameter
                )

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                semantic_embeddings.extend(batch_embeddings)

                # Update Azure metrics
                azure_metrics["successful_requests"] += 1
                azure_metrics["total_time"] += time.time() - start_time

                # Reset circuit breaker failures on success
                circuit_failures = 0

            except Exception as e:
                error_msg = str(e).lower()

                # Handle rate limiting with proper Azure backoff
                if "rate limit" in error_msg or "too many requests" in error_msg or "throttl" in error_msg:
                    azure_metrics["rate_limited_requests"] += 1
                    wait_time = min(60, 2 ** (azure_metrics["rate_limited_requests"] % 6))
                    logging.warning(f"Azure OpenAI rate limit hit (ID: {correlation_id}). Backing off for {wait_time}s")
                    time.sleep(wait_time)

                    # Try again with the same batch (don't increment i)
                    i -= batch_size
                    continue

                # Handle token limit errors by reducing batch size
                elif "token" in error_msg or "context length" in error_msg:
                    if batch_size > 1:
                        # Reduce batch size and retry with smaller batches
                        new_batch_size = max(1, batch_size // 2)
                        logging.warning(f"Token limit exceeded. Reducing batch size from {batch_size} to {new_batch_size}")
                        batch_size = new_batch_size
                        total_batches = (len(text_series) - i + batch_size - 1) // batch_size

                        # Try again with the same starting point but smaller batch
                        i -= batch_size * 2  # Go back to retry with smaller batch
                        i = max(0, i)  # Ensure we don't go negative
                        continue
                    else:
                        # Can't reduce batch size further, use zeros for this item
                        logging.error(f"Single item is too long: {len(text_series.iloc[i])}")
                        semantic_embeddings.extend([[0.0] * 3072])
                        azure_metrics["failed_requests"] += 1

                # Other errors - implement circuit breaker pattern
                else:
                    azure_metrics["failed_requests"] += 1
                    circuit_failures += 1
                    logging.error(f"Embedding error: {error_msg} (Failures: {circuit_failures}/{circuit_threshold})")

                    # If too many failures, open circuit breaker
                    if circuit_failures >= circuit_threshold:
                        circuit_open = True
                        circuit_reset_time = time.time()
                        logging.error(f"Circuit breaker activated! Too many failures. Pausing API calls for 60s")

                    # Use zero vector for errors
                    semantic_embeddings.extend([[0.0] * 3072 for _ in range(batch_size_actual)])

            # Update progress
            pbar.update(batch_size_actual)

    # Log Azure metrics
    avg_time = azure_metrics["total_time"] / max(1, azure_metrics["successful_requests"])
    logging.info(f"Azure OpenAI Embedding API metrics: {azure_metrics['total_requests']} requests, "
                f"{azure_metrics['successful_requests']} successful, {azure_metrics['failed_requests']} failed, "
                f"{azure_metrics['rate_limited_requests']} rate limited, avg time: {avg_time:.2f}s")

    # Ensure we return the right number of embeddings
    if len(semantic_embeddings) < len(text_series):
        logging.warning(f"Missing embeddings: expected {len(text_series)}, got {len(semantic_embeddings)}. Padding with zeros.")
        semantic_embeddings.extend([[0.0] * 3072 for _ in range(len(text_series) - len(semantic_embeddings))])

    return np.array(semantic_embeddings)


def process_embedding_batch(batch, model, retry_limit=3):
    """Process a single batch of embeddings with retries and proper error handling"""
    embedding_dim = 3072  # text-embedding-3-large dimension

    for retry in range(retry_limit):
        try:
            # Call the embedding API with original text (no truncation)
            response = embedding_client.embeddings.create(
                input=batch,
                model=model,
                timeout=30
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            error_msg = str(e)

            # Handle token limit errors by reducing batch size
            if "maximum context length" in error_msg.lower() or "token" in error_msg.lower():
                if retry < retry_limit - 1:
                    logging.warning(f"Token limit exceeded. Trying with smaller batch size...")

                    # If possible, split the batch in half and process recursively
                    if len(batch) > 1:
                        logging.info(f"Splitting batch of size {len(batch)} into smaller chunks")
                        mid = len(batch) // 2
                        first_half = process_embedding_batch(batch[:mid], model, retry_limit)
                        second_half = process_embedding_batch(batch[mid:], model, retry_limit)
                        return first_half + second_half
                    else:
                        # If we can't split further, log the issue
                        logging.error(f"Single item is too long for embedding API. Consider preprocessing text to reduce length.")
                        return [[0.0] * embedding_dim]  # Return zero vector for this item
                else:
                    logging.error(f"Token limit still exceeded after multiple attempts. Using fallback embeddings.")
                    return [[0.0] * embedding_dim for _ in range(len(batch))]

            # Handle rate limiting with exponential backoff
            elif "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                wait_time = 2 ** retry * 2
                logging.warning(f"Rate limit hit, backing off for {wait_time}s")
                time.sleep(wait_time)

            # Handle other errors
            elif retry < retry_limit - 1:
                wait_time = 2 ** retry
                logging.error(f"Embedding error on attempt {retry+1}/{retry_limit}: {error_msg}")
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"All embedding retries failed: {error_msg}")
                # Return zero vectors as fallback
                return [[0.0] * embedding_dim for _ in range(len(batch))]

    # Should never reach here due to the handling in the loop, but just in case
    return [[0.0] * embedding_dim for _ in range(len(batch))]

# %%
def apply_labels_to_data(df, labeled_clusters, domains):
    """Apply cluster labels and domains to the data"""

    result_df = df.copy()

    # Create mapping dictionaries for efficient lookups
    topic_mapping = {int(k): v["topic"] for k, v in labeled_clusters.items() if k != "-1"}
    topic_mapping[-1] = "Noise"  # Add mapping for noise cluster

    # Create domain mapping
    domain_mapping = {}
    for domain in domains.get("domains", []):
        domain_name = domain["domain_name"]
        for cluster_id in domain.get("clusters", []):
            domain_mapping[int(cluster_id)] = domain_name

    # Apply mappings to create category and subcategory columns
    result_df["subcategory"] = result_df["cluster"].map(topic_mapping).fillna("Unknown")
    result_df["category"] = result_df["cluster"].map(domain_mapping).fillna("Other")

    return result_df

# %%
def hybrid_embedding_hdbscan_clustering(
    df,
    embedding_col='embedding',
    min_cluster_size=25,
    min_samples=5,
    umap_n_components=50,
    umap_n_neighbors=100,
    umap_min_dist=0.03,
    metric='cosine',
    random_state=42
):
    """Apply UMAP + HDBSCAN clustering to hybrid embeddings"""

    start_time = time.time()
    logging.info(f"Starting HDBSCAN clustering for {len(df)} documents")
    result_df = df.copy()

    # Extract embeddings from the dataframe
    logging.info("Extracting embeddings...")
    embeddings = np.array([json.loads(emb) for emb in df[embedding_col].tolist()])

    # Standardize embeddings
    logging.info("Standardizing embeddings...")
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Apply UMAP for dimensionality reduction
    logging.info(f"Applying UMAP to reduce dimensions from {scaled_embeddings.shape[1]} to {umap_n_components}...")
    reducer = umap.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=True,
        spread=1.0,
        local_connectivity=2,
        verbose=True
    )

    umap_embeddings = reducer.fit_transform(scaled_embeddings)

    # Apply HDBSCAN for clustering
    logging.info(f"Applying HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='leaf',
        cluster_selection_epsilon=0.2,
        prediction_data=True,
        core_dist_n_jobs=-1
    )

    cluster_labels = clusterer.fit_predict(umap_embeddings)

    # Add cluster labels to dataframe
    result_df['cluster'] = cluster_labels

    # Add cluster probabilities
    if hasattr(clusterer, 'probabilities_'):
        result_df['cluster_probability'] = clusterer.probabilities_

    # Count clusters and noise points
    clusters = set(cluster_labels)
    n_clusters = len(clusters) - (1 if -1 in clusters else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_percentage = 100 * n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0

    logging.info(f"HDBSCAN found {n_clusters} clusters")
    logging.info(f"Noise points: {n_noise} ({noise_percentage:.2f}% of data)")

    # Save model artifacts locally
    with open("results/dataset/umap_reducer.pkl", "wb") as f:
        import pickle
        pickle.dump(reducer, f)

    with open("results/dataset/hdbscan_clusterer.pkl", "wb") as f:
        import pickle
        pickle.dump(clusterer, f)

    total_time = time.time() - start_time
    logging.info(f"Total HDBSCAN processing time: {total_time:.2f} seconds")
    return result_df, umap_embeddings, clusterer, reducer

# %%
def group_clusters_into_domains(labeled_clusters, clusters_info, umap_embeddings, cluster_labels,
                               max_domains=20, output_dir=None):
    """
    Group clusters into domains using coordinate proximity and semantic similarity
    with automatic domain count optimization and empty domain removal

    Args:
        labeled_clusters: Dictionary of cluster labels from label_clusters_with_llm
        clusters_info: Dictionary of cluster information
        umap_embeddings: UMAP reduced embeddings
        cluster_labels: Array of cluster assignments
        max_domains: Maximum number of domains to consider (upper bound)
        output_dir: Directory to save results

    Returns:
        Dictionary with domain groupings and standardized labels
    """
    import uuid
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    logging.info(f"Grouping clusters into domains with auto-optimization (max: {max_domains} domains)")

    # Skip noise cluster (-1) for domain grouping
    clusters_to_group = {k: v for k, v in labeled_clusters.items() if k != "-1"}

    # If very few clusters, don't bother with domains
    if len(clusters_to_group) <= 3:
        domains = {
            "domains": [
                {
                    "domain_name": "All Incidents",
                    "description": "All incident clusters",
                    "clusters": [int(k) for k in clusters_to_group.keys()]
                }
            ]
        }
        domains["domains"].append({"domain_name": "Noise", "description": "Uncategorized incidents", "clusters": [-1]})

        if output_dir:
            with open(f"{output_dir}/domains.json", "w") as f:
                json.dump(domains, f, indent=2)

        return domains

    # 1. Calculate cluster centroids in UMAP space
    unique_clusters = np.unique(cluster_labels)
    cluster_centroids = {}

    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise
            continue
        # Get points belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_points = umap_embeddings[cluster_mask]
        # Calculate centroid (if cluster has points)
        if len(cluster_points) > 0:
            cluster_centroids[cluster_id] = np.mean(cluster_points, axis=0)

    # 2. Convert centroids to array for hierarchical clustering
    centroid_ids = list(cluster_centroids.keys())
    centroid_coords = np.array([cluster_centroids[cid] for cid in centroid_ids])

    # 3. Determine optimal number of domains using internal metrics
    max_domains_to_try = min(max_domains, len(centroid_ids) - 1)  # Don't try more domains than clusters - 1
    max_domains_to_try = max(2, max_domains_to_try)  # Ensure at least 2 domains to compare

    scores = []
    domain_counts = range(2, max_domains_to_try + 1)

    logging.info("Determining optimal number of domains...")
    for n_domains in domain_counts:
        # Use hierarchical clustering with the current domain count
        hc = AgglomerativeClustering(n_clusters=n_domains, linkage='ward')
        domain_labels = hc.fit_predict(centroid_coords)

        # Calculate silhouette score for this clustering
        if len(np.unique(domain_labels)) > 1:  # Ensure at least 2 domains for score calculation
            try:
                sil_score = silhouette_score(centroid_coords, domain_labels)
                ch_score = calinski_harabasz_score(centroid_coords, domain_labels)
                # Combine metrics (normalize CH score as it can be very large)
                combined_score = sil_score + (ch_score / (1000 + ch_score))
                scores.append(combined_score)
                logging.info(f"  {n_domains} domains: silhouette={sil_score:.3f}, CH={ch_score:.1f}, combined={combined_score:.3f}")
            except Exception as e:
                logging.warning(f"Error calculating score for {n_domains} domains: {e}")
                scores.append(-1)  # Mark as invalid
        else:
            scores.append(-1)  # Invalid score

    # Find optimal number of domains based on scores
    valid_scores = [(i, score) for i, score in enumerate(scores) if score > 0]
    if valid_scores:
        # Get the index of the highest score
        best_idx, best_score = max(valid_scores, key=lambda x: x[1])
        optimal_domains = domain_counts[best_idx]
        logging.info(f"Selected optimal domain count: {optimal_domains} (score: {best_score:.3f})")
    else:
        # Fallback to a reasonable default
        optimal_domains = min(15, max(2, len(centroid_ids) // 5))
        logging.warning(f"Could not determine optimal domains. Using default: {optimal_domains}")

    # 4. Apply hierarchical clustering with optimal domain count
    hc = AgglomerativeClustering(n_clusters=optimal_domains, linkage='ward')
    domain_labels = hc.fit_predict(centroid_coords)

    # 5. Create initial domain mapping
    initial_domains = {}
    for i, domain_id in enumerate(domain_labels):
        if domain_id not in initial_domains:
            initial_domains[domain_id] = []
        initial_domains[domain_id].append(centroid_ids[i])

    # 6. Remove empty domains (although they shouldn't exist at this point)
    initial_domains = {k: v for k, v in initial_domains.items() if v}
    domain_count = len(initial_domains)

    logging.info(f"Initial domain grouping created {domain_count} domains based on cluster coordinates")

    # 7. Use LLM to name and refine domains, and standardize labels WITHIN each domain
    final_domains = []
    all_standardized_labels = {}

    # Process each domain with its own LLM call for naming and label standardization
    for domain_id, cluster_ids in initial_domains.items():
        # [existing domain processing code remains the same]
        # This part involves sending the domain information to the LLM for naming and standardization
        try:
            # Create prompt for this domain
            prompt_text = f"I have a group of related IT incident clusters that form a domain.\n\n"
            prompt_text += "CLUSTERS IN THIS DOMAIN:\n\n"

            # Add each cluster's info to the prompt
            for cluster_id in cluster_ids:
                str_cluster_id = str(cluster_id)
                if str_cluster_id in clusters_info:
                    size = clusters_info[str_cluster_id]["size"]
                    percentage = clusters_info[str_cluster_id]["percentage"]
                else:
                    size = 0
                    percentage = 0

                if str_cluster_id in labeled_clusters:
                    topic = labeled_clusters[str_cluster_id]["topic"]
                    description = labeled_clusters[str_cluster_id]["description"]
                else:
                    topic = f"Cluster {cluster_id}"
                    description = "No description available"

                prompt_text += f"\nCluster {cluster_id} (Size: {size}, {percentage}%):\n"
                prompt_text += f"Current Topic Label: {topic}\n"
                prompt_text += f"Description: {description}\n"

            # Request for domain naming and label standardization
            prompt_text += f"""
Based on these related clusters, please:
1. Provide an appropriate name for this domain of incidents
2. Write a brief description of what unifies these incidents
3. Standardize the topic labels to use consistent terminology within this domain

YOU MUST RETURN A VALID JSON OBJECT with this exact structure:
{{
  "domain_name": "Descriptive Domain Name",
  "description": "Brief description of what unifies these incidents",
  "standardized_labels": {{
    "{cluster_ids[0]}": "Standardized topic label for cluster {cluster_ids[0]}",
    "{cluster_ids[1]}": "Standardized topic label for cluster {cluster_ids[1]}",
    ... for all clusters in the domain
  }}
}}

Ensure standardized labels maintain specific system names where present but use consistent terminology.
"""

            # Generate a correlation ID for Azure tracking
            correlation_id = str(uuid.uuid4())

            # Call Azure OpenAI with retry logic
            retry_attempts = 3
            domain_result = None

            for attempt in range(retry_attempts):
                try:
                    logging.info(f"Processing domain {domain_id+1}/{domain_count} with {len(cluster_ids)} clusters (attempt {attempt+1})")

                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an IT domain classification expert. Return properly structured JSON with no additional text."},
                            {"role": "user", "content": prompt_text}
                        ],
                        temperature=0.3,
                        response_format={"type": "json_object"},
                        timeout=30,
                        user=correlation_id
                    )

                    # Parse response
                    result_text = response.choices[0].message.content
                    domain_result = robust_json_parser(result_text)

                    # Validate expected structure
                    if "domain_name" in domain_result and "description" in domain_result and "standardized_labels" in domain_result:
                        break
                    else:
                        missing = []
                        if "domain_name" not in domain_result: missing.append("domain_name")
                        if "description" not in domain_result: missing.append("description")
                        if "standardized_labels" not in domain_result: missing.append("standardized_labels")
                        raise ValueError(f"Response missing required keys: {missing}")

                except Exception as e:
                    if attempt < retry_attempts - 1:
                        wait_time = min(30, 2 ** attempt * 2)
                        logging.warning(f"Domain processing attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        # Use fallback for this domain if all attempts fail
                        logging.error(f"Domain processing failed after all attempts: {e}")
                        sample_cluster = str(cluster_ids[0]) if cluster_ids else "unknown"
                        sample_topic = labeled_clusters.get(sample_cluster, {}).get("topic", f"Domain {domain_id}")
                        domain_result = {
                            "domain_name": f"Domain {domain_id}: {sample_topic}",
                            "description": "Group of related incidents",
                            "standardized_labels": {
                                str(cid): labeled_clusters.get(str(cid), {}).get("topic", f"Cluster {cid}")
                                for cid in cluster_ids
                            }
                        }

            # Add the domain to our results (only if it has clusters)
            if domain_result and cluster_ids:
                # Convert cluster IDs from strings to integers for the domain definition
                final_domains.append({
                    "domain_name": domain_result["domain_name"],
                    "description": domain_result["description"],
                    "clusters": [int(cid) for cid in cluster_ids]
                })

                # Add standardized labels to the overall collection
                if "standardized_labels" in domain_result:
                    for cluster_id, new_label in domain_result["standardized_labels"].items():
                        all_standardized_labels[cluster_id] = new_label

        except Exception as e:
            logging.error(f"Error processing domain {domain_id}: {e}")
            # Skip this domain if it has no clusters
            if not cluster_ids:
                continue

            # Add a fallback domain
            final_domains.append({
                "domain_name": f"Domain {domain_id}",
                "description": "Group of related incidents",
                "clusters": [int(cid) for cid in cluster_ids]
            })

    # Add the noise domain
    final_domains.append({
        "domain_name": "Noise",
        "description": "Uncategorized incidents",
        "clusters": [-1]
    })

    # Create the final result structure
    domains = {"domains": final_domains}

    # Apply standardized labels to the labeled_clusters dictionary
    for cluster_id, new_label in all_standardized_labels.items():
        if cluster_id in labeled_clusters:
            labeled_clusters[cluster_id]["topic"] = new_label

    # Save updated labeled clusters if we made changes
    if all_standardized_labels and output_dir:
        with open(f"{output_dir}/labeled_clusters_standardized.json", "w") as f:
            json.dump(labeled_clusters, f, indent=2)

    # Save domain metrics for reference
    if output_dir:
        metrics = {
            "optimal_domain_count": optimal_domains,
            "tested_domain_counts": list(domain_counts),
            "domain_scores": [float(s) if s > 0 else None for s in scores],
            "final_domain_count": len(final_domains) - 1  # Exclude noise domain
        }
        with open(f"{output_dir}/domain_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Save domains
    if output_dir:
        with open(f"{output_dir}/domains.json", "w") as f:
            json.dump(domains, f, indent=2)

    return domains

# %%
def generate_embeddings_pipeline(input_query, dataset_name, embeddings_table_id=None,
                                batch_size=25, save_to_bq=True, summary_path=None,
                                write_disposition="WRITE_APPEND", result_path="results"):
    """
    First pipeline stage: Generate hybrid embeddings from text data with performance optimizations

    Args:
        input_query: BigQuery query to load data
        dataset_name: Name for the dataset/run
        embeddings_table_id: BigQuery table to save embeddings
        batch_size: Size of batches for embedding API calls
        save_to_bq: Whether to save embeddings to BigQuery
        summary_path: Path to precomputed summaries to skip LLM processing
        write_disposition: BigQuery write disposition
        result_path: Base path to store results

    Returns:
        Tuple of (DataFrame with embeddings, classification result, fallback statistics)
    """
    start_time = time.time()

    # Create output directory
    output_dir = f"{result_path}/{dataset_name}/embeddings"
    intermediate_dir = f"{result_path}/{dataset_name}/intermediate"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    logging.info(f"Generating embeddings for dataset '{dataset_name}'")

    # Check if we should use previously computed summaries
    if not summary_path:
        # Look for summaries in the intermediate directory
        default_summary_path = f"{intermediate_dir}/df_with_summaries.parquet"
        if os.path.exists(default_summary_path):
            logging.info(f"Found precomputed summaries at {default_summary_path}")
            summary_path = default_summary_path

    # 1. Load data from BigQuery
    logging.info(f"Loading data with query: {input_query}")
    df = run_query(input_query)
    logging.info(f"Loaded {len(df)} records")

    # Save raw data for later use
    df.to_parquet(f"{output_dir}/raw_data.parquet", index=False)

    # Log Azure telemetry
    perf_metrics = {
        "start_time": datetime.now().isoformat(),
        "record_count": len(df),
        "batch_size": batch_size,
        "dataset_name": dataset_name,
        "using_precomputed_summaries": summary_path is not None
    }

    # 2. Generate hybrid embeddings with optimized batch processing
    df_with_embeddings, classification_result, fallback_stats = create_hybrid_embeddings(
        df,
        batch_size=batch_size,
        summary_path=summary_path,
        result_path=result_path,
        dataset_name=dataset_name
    )

    # Save embeddings to file
    df_with_embeddings.to_parquet(f"{output_dir}/df_with_embeddings.parquet", index=False)

    # Save entity and action classification
    with open(f"{output_dir}/classification_result.json", "w") as f:
        json.dump(classification_result, f, indent=2, cls=NumpyEncoder)

    # Save fallback statistics to file
    with open(f"{output_dir}/fallback_stats.json", "w") as f:
        json.dump(fallback_stats, f, indent=2, cls=NumpyEncoder)

    # Add fallback statistics to performance metrics if available
    if "precomputed" not in fallback_stats:
        perf_metrics["text_summarization"] = {
            "llm_processed": fallback_stats["llm_processed"],
            "llm_success_count": fallback_stats["llm_success_count"],
            "short_desc_fallbacks": fallback_stats["short_desc_fallbacks"],
            "api_failure_fallbacks": fallback_stats["api_failure_fallbacks"],
            "final_sweep_fallbacks": fallback_stats["final_sweep_fallbacks"],
            "llm_success_rate": fallback_stats["llm_success_rate"],
            "overall_llm_rate": fallback_stats["overall_llm_rate"]
        }

    # 3. Optionally save embeddings to BigQuery
    if save_to_bq and embeddings_table_id:
        save_embeddings_to_bigquery(df_with_embeddings, embeddings_table_id, write_disposition)

    # Log runtime
    total_time = time.time() - start_time
    logging.info(f"Embeddings generation completed in {total_time:.2f} seconds")

    # Update and save performance metrics
    perf_metrics.update({
        "end_time": datetime.now().isoformat(),
        "runtime_seconds": total_time,
        "avg_time_per_record": total_time / max(1, len(df)),
        "records_per_second": len(df) / max(1, total_time)
    })

    with open(f"{output_dir}/embedding_metadata.json", "w") as f:
        json.dump(perf_metrics, f, indent=2)

    return df_with_embeddings, classification_result, fallback_stats

# %%
def read_parquet_optimized(file_path, columns=None, chunksize=None):
    """
    Read a Parquet file with memory optimization options

    Args:
        file_path: Path to the Parquet file
        columns: List of columns to load (None loads all)
        chunksize: Number of rows to read at once (None reads all at once)

    Returns:
        DataFrame with the loaded data
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    try:
        # First approach: Try with memory_map=True which may reduce memory usage
        try:
            if columns is not None:
                return pd.read_parquet(file_path, columns=columns, memory_map=True)
            else:
                return pd.read_parquet(file_path, memory_map=True)
        except Exception as e:
            logging.warning(f"Memory-mapped read failed: {e}, trying with chunking")

        # Second approach: Read in chunks
        if chunksize is not None:
            # Open the Parquet file
            parquet_file = pq.ParquetFile(file_path)

            # Get total number of row groups
            total_row_groups = parquet_file.num_row_groups

            # Initialize an empty list to store dataframes
            chunks = []

            # Read row groups in chunks
            for i in range(0, total_row_groups, chunksize):
                # Calculate the end range (not exceeding total_row_groups)
                end = min(i + chunksize, total_row_groups)

                # Read specific row groups
                table = parquet_file.read_row_groups(list(range(i, end)), columns=columns)

                # Convert to pandas dataframe
                df_chunk = table.to_pandas()
                chunks.append(df_chunk)

            # Combine all chunks
            return pd.concat(chunks, ignore_index=True)

        # Third approach: Try with default options but limit columns
        if columns is not None:
            return pd.read_parquet(file_path, columns=columns)
        else:
            # Last resort - try to read with default settings
            return pd.read_parquet(file_path)

    except Exception as e:
        logging.error(f"Error reading Parquet file: {e}")
        # If we still have issues, try an even more memory efficient approach
        try:
            # Read only essential columns for clustering
            essential_cols = ['number', 'short_description', 'cluster', 'cluster_probability']
            if columns is not None:
                essential_cols = [col for col in essential_cols if col in columns]

            logging.info(f"Attempting to read only essential columns: {essential_cols}")
            return pd.read_parquet(file_path, columns=essential_cols)
        except Exception as e2:
            logging.error(f"Failed to read even with essential columns: {e2}")
            raise

# %%
def train_hdbscan_pipeline(df_with_embeddings=None, dataset_name=None, embedding_path=None,
                          min_cluster_size=25, min_samples=5, umap_n_components=50,
                          umap_n_neighbors=100, umap_min_dist=0.1, metric='cosine',
                          random_state=42, result_path="results", use_checkpoint=True):
    """
    Second pipeline stage: Train HDBSCAN clustering model on embeddings
    with UMAP checkpoint for improved performance

    Args:
        df_with_embeddings: DataFrame with 'embedding' column
        dataset_name: Name for the dataset/run
        embedding_path: Optional custom path to load embeddings from
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        umap_n_components: Number of dimensions for UMAP reduction
        umap_n_neighbors: Number of neighbors for UMAP
        umap_min_dist: Minimum distance for UMAP
        metric: Distance metric for UMAP
        random_state: Random seed for reproducibility
        result_path: Base path to store results locally (default: "results")
        use_checkpoint: Whether to use checkpointing for UMAP embeddings (default: True)

    Returns:
        Tuple of (clustered_df, umap_embeddings, clusterer, reducer)
    """
    start_time = time.time()
    correlation_id = str(uuid.uuid4())  # Generate a correlation ID for tracking

    # Check if we need to load data from file
    if df_with_embeddings is None:
        # If custom embedding path is provided, use it directly
        if embedding_path:
            if not os.path.exists(embedding_path):
                raise FileNotFoundError(f"Embeddings file not found at specified path: {embedding_path}")

            logging.info(f"Loading embeddings from custom path: {embedding_path}")
            df_with_embeddings = pd.read_parquet(embedding_path)

        # Otherwise use the standard path constructed from dataset_name
        elif dataset_name:
            default_path = f"{result_path}/{dataset_name}/embeddings/df_with_embeddings.parquet"
            if not os.path.exists(default_path):
                raise FileNotFoundError(f"Embeddings file not found at default path: {default_path}")

            logging.info(f"Loading embeddings from default path: {default_path}")
            df_with_embeddings = pd.read_parquet(default_path)
        else:
            raise ValueError("Either df_with_embeddings, dataset_name, or embedding_path must be provided")

    # Create output directory for local saving
    output_dir = f"{result_path}/{dataset_name}/clustering" if dataset_name else f"{result_path}/clustering"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Training HDBSCAN for dataset '{dataset_name or 'custom'}'")

    # Define Azure blob paths
    azure_umap_blob_path = f"{dataset_name}/umap_reducer.pkl"
    azure_hdbscan_blob_path = f"{dataset_name}/hdbscan_clusterer.pkl"
    azure_metadata_blob_path = f"{dataset_name}/clustering_metadata.json"

    # Initialize tracking for successful uploads
    azure_uploads = {
        "successful": [],
        "failed": []
    }

    # Define path for UMAP checkpoint
    umap_checkpoint_path = f"{output_dir}/umap_embeddings_checkpoint.npy"
    umap_reducer_checkpoint_path = f"{output_dir}/umap_reducer_checkpoint.pkl"

    # Check if UMAP checkpoint exists and should be used
    umap_embeddings = None
    reducer = None
    use_existing_checkpoint = use_checkpoint and os.path.exists(umap_checkpoint_path) and os.path.exists(umap_reducer_checkpoint_path)

    if use_existing_checkpoint:
        try:
            logging.info(f"Found UMAP checkpoint. Loading embeddings from {umap_checkpoint_path}")
            umap_embeddings = np.load(umap_checkpoint_path)

            logging.info(f"Loading UMAP reducer from {umap_reducer_checkpoint_path}")
            with open(umap_reducer_checkpoint_path, 'rb') as f:
                reducer = pickle.load(f)

            logging.info(f"Successfully loaded UMAP checkpoint with {umap_embeddings.shape[1]} dimensions")
        except Exception as e:
            logging.error(f"Error loading UMAP checkpoint: {e}. Will recompute UMAP projection.")
            umap_embeddings = None
            reducer = None

    # If no checkpoint was loaded, run the UMAP projection
    if umap_embeddings is None:
        # Extract embeddings from the dataframe
        logging.info("Extracting embeddings...")
        embeddings = np.array([json.loads(emb) for emb in df_with_embeddings['embedding'].tolist()])

        # Standardize embeddings
        logging.info("Standardizing embeddings...")
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)

        # Apply UMAP for dimensionality reduction
        logging.info(f"Applying UMAP to reduce dimensions from {scaled_embeddings.shape[1]} to {umap_n_components}...")
        reducer = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=metric,
            random_state=random_state,
            low_memory=True,
            spread=1.0,
            local_connectivity=2,
            verbose=True
        )

        umap_embeddings = reducer.fit_transform(scaled_embeddings)

        # Save checkpoint if checkpointing is enabled
        if use_checkpoint:
            try:
                logging.info(f"Saving UMAP embeddings checkpoint to {umap_checkpoint_path}")
                np.save(umap_checkpoint_path, umap_embeddings)

                logging.info(f"Saving UMAP reducer to {umap_reducer_checkpoint_path}")
                with open(umap_reducer_checkpoint_path, 'wb') as f:
                    pickle.dump(reducer, f)

                logging.info("UMAP checkpoint saved successfully")
            except Exception as e:
                logging.error(f"Error saving UMAP checkpoint: {e}")

    # Apply HDBSCAN for clustering (this runs whether we loaded from checkpoint or not)
    logging.info(f"Applying HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='leaf',
        cluster_selection_epsilon=0.2,
        prediction_data=True,
        core_dist_n_jobs=-1
    )

    cluster_labels = clusterer.fit_predict(umap_embeddings)

    # Rest of the function remains unchanged...
    # Add cluster labels to dataframe
    result_df = df_with_embeddings.copy()
    result_df['cluster'] = cluster_labels

    # Add cluster probabilities
    if hasattr(clusterer, 'probabilities_'):
        result_df['cluster_probability'] = clusterer.probabilities_

    # Count clusters and noise points
    clusters = set(cluster_labels)
    n_clusters = len(clusters) - (1 if -1 in clusters else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_percentage = 100 * n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0

    logging.info(f"HDBSCAN found {n_clusters} clusters")
    logging.info(f"Noise points: {n_noise} ({noise_percentage:.2f}% of data)")

    # Save clustered data locally
    result_df.to_parquet(f"{output_dir}/clustered_df.parquet", index=False)

    # Save UMAP embeddings for subsequent domain grouping
    try:
        np.save(f"{output_dir}/umap_embeddings.npy", umap_embeddings)
    except Exception as e:
        logging.warning(f"Could not save final UMAP embeddings as numpy array: {e}")

    # Save UMAP reducer and HDBSCAN clusterer
    with open(f"{output_dir}/umap_reducer.pkl", "wb") as f:
        pickle.dump(reducer, f)

    with open(f"{output_dir}/hdbscan_clusterer.pkl", "wb") as f:
        pickle.dump(clusterer, f)

    # Create metadata object including checkpoint info
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "umap_n_components": umap_n_components,
            "umap_n_neighbors": umap_n_neighbors,
            "umap_min_dist": umap_min_dist,
            "metric": metric,
            "random_state": random_state
        },
        "results": {
            "num_clusters": n_clusters,
            "noise_points": n_noise,
            "noise_percentage": noise_percentage
        },
        "runtime_seconds": time.time() - start_time,
        "dataset_name": dataset_name,
        "correlation_id": correlation_id,
        "checkpoint_used": use_existing_checkpoint
    }

    # Save runtime metadata locally
    with open(f"{output_dir}/clustering_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)

    # Upload artifacts to Azure (unchanged)
    # [... upload code remains the same ...]

    # Calculate total runtime
    total_time = time.time() - start_time
    logging.info(f"HDBSCAN training completed in {total_time:.2f} seconds")

    return result_df, umap_embeddings, clusterer, reducer

# %%
def generate_cluster_info(df, text_column='short_description', cluster_column='cluster', sample_size=5, output_dir=None):
    """
    Generate detailed information about each cluster including size, percentage, and sample texts

    Args:
        df: DataFrame with clustering results
        text_column: Column containing text data
        cluster_column: Column containing cluster labels
        sample_size: Number of random samples to include for each cluster
        output_dir: Directory to save cluster details

    Returns:
        Dictionary with cluster information
    """
    logging.info(f"Generating cluster information from {len(df)} records")

    # Count instances of each cluster
    cluster_counts = df[cluster_column].value_counts().to_dict()
    total_records = len(df)

    # Initialize results dictionary
    clusters_info = {}

    # Process each cluster
    for cluster_id, count in cluster_counts.items():
        # Calculate percentage
        percentage = round((count / total_records) * 100, 2)

        # Get samples
        cluster_mask = df[cluster_column] == cluster_id
        samples = df[cluster_mask].sample(min(sample_size, count))[text_column].tolist()

        # Store information
        cluster_key = str(cluster_id)  # Convert to string for JSON compatibility
        clusters_info[cluster_key] = {
            "size": int(count),
            "percentage": percentage,
            "samples": samples
        }

    # Save to file if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/cluster_details.json", "w") as f:
            json.dump(clusters_info, f, indent=2, cls=NumpyEncoder)
        logging.info(f"Saved cluster details to {output_dir}/cluster_details.json")

    return clusters_info

# %%
def label_clusters_with_llm(clusters_info, max_samples=300, chunk_size=25, output_dir=None):
    """
    Use Azure OpenAI to generate meaningful labels and descriptions for clusters

    Args:
        clusters_info: Dictionary with cluster information
        max_samples: Maximum number of text samples to include per cluster
        chunk_size: Number of clusters to process in each LLM batch
        output_dir: Directory to save labeled clusters

    Returns:
        Dictionary with labeled clusters
    """
    logging.info(f"Generating cluster labels with LLM for {len(clusters_info)} clusters")

    # Skip noise cluster (ID -1) initially and add it back at the end
    noise_cluster = clusters_info.get("-1", None)
    clusters_to_label = {k: v for k, v in clusters_info.items() if k != "-1"}

    if not clusters_to_label:
        logging.warning("No clusters to label (excluding noise)")
        return {"-1": {"topic": "Noise", "description": "Uncategorized data points"}}

    # Process clusters in chunks to avoid token limits
    cluster_ids = list(clusters_to_label.keys())
    labeled_clusters = {}

    for i in range(0, len(cluster_ids), chunk_size):
        batch_ids = cluster_ids[i:i+chunk_size]
        logging.info(f"Labeling clusters {i} to {i+len(batch_ids)-1}")

        # Create prompt for this batch
        prompt = "Analyze these clusters of IT incidents and provide a concise topic name and description for each.\n\n"

        for cluster_id in batch_ids:
            cluster = clusters_to_label[cluster_id]
            # Get samples, limiting to max_samples
            samples = cluster["samples"][:min(len(cluster["samples"]), max_samples)]
            size = cluster["size"]
            percentage = cluster["percentage"]

            prompt += f"CLUSTER {cluster_id} ({size} incidents, {percentage}% of total):\n"
            prompt += "SAMPLES:\n"
            for j, sample in enumerate(samples):
                prompt += f"{j+1}. {sample}\n"
            prompt += "\n"

        prompt += """For each cluster above, provide:
1. A concise topic name (3-7 words) that captures the specific system and issue
2. A brief description (1-2 sentences) that explains the common theme

Format your response as a valid JSON object with this structure:
{
  "cluster_id_1": {
    "topic": "Concise topic name",
    "description": "Brief description of the cluster theme"
  },
  "cluster_id_2": {
    "topic": "Another topic name",
    "description": "Description of another cluster theme"
  }
}

Make your topics specific and technical. Include the affected system name (like SAP, Outlook, VPN) and the specific issue or error where identifiable.
"""

        # Generate a correlation ID for Azure tracking
        correlation_id = str(uuid.uuid4())

        # Call Azure OpenAI with retry logic
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                logging.info(f"Sending batch of {len(batch_ids)} clusters to LLM (attempt {attempt+1})")

                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert IT analyst that provides concise, specific labels for clusters of IT incidents. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    timeout=60,
                    user=correlation_id
                )

                # Parse response
                result_text = response.choices[0].message.content
                batch_labels = robust_json_parser(result_text)

                # Add to main results
                if batch_labels:
                    labeled_clusters.update(batch_labels)
                    break
                else:
                    raise ValueError("Empty response from LLM")

            except Exception as e:
                if attempt < retry_attempts - 1:
                    wait_time = min(60, 2 ** (attempt + 1))
                    logging.warning(f"Cluster labeling attempt {attempt+1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All cluster labeling attempts failed for batch {i}: {e}")

                    # Create fallback labels for this batch
                    for cluster_id in batch_ids:
                        if cluster_id not in labeled_clusters:
                            labeled_clusters[cluster_id] = {
                                "topic": f"Cluster {cluster_id}",
                                "description": "Group of related IT incidents"
                            }

    # Add noise cluster label
    labeled_clusters["-1"] = {"topic": "Noise", "description": "Uncategorized data points"}

    # Save to file if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/labeled_clusters.json", "w") as f:
            json.dump(labeled_clusters, f, indent=2, cls=NumpyEncoder)
        logging.info(f"Saved labeled clusters to {output_dir}/labeled_clusters.json")

    return labeled_clusters

# %%
# def analyze_clusters_pipeline(clustered_df=None, dataset_name=None, num_domains=15, result_path="results"):
#     """
#     Third pipeline stage: Generate cluster info, label clusters and group into domains
#     """
#     start_time = time.time()

#     # Check if we need to load data from file
#     if clustered_df is None:
#         if dataset_name is None:
#             raise ValueError("Either clustered_df or dataset_name must be provided")

#         # Load clustering results from file
#         cluster_path = f"{result_path}/{dataset_name}/clustering/clustered_df.parquet"
#         if not os.path.exists(cluster_path):
#             raise FileNotFoundError(f"Clustered data not found at {cluster_path}")

#         logging.info(f"Loading clustered data from {cluster_path} with memory optimization")
#         try:
#             # First try to load only essential columns for clustering
#             essential_cols = ['number', 'short_description', 'cluster', 'cluster_probability', 'embedding']
#             clustered_df = read_parquet_optimized(
#                 cluster_path,
#                 columns=essential_cols,
#                 chunksize=10  # Read 10 row groups at a time
#             )
#             logging.info(f"Loaded clustered data with {len(clustered_df)} rows and {len(clustered_df.columns)} columns")
#         except Exception as e:
#             logging.error(f"Error loading clustered data: {e}")
#             raise

#     # Create output directory
#     output_dir = f"{result_path}/{dataset_name}/analysis"
#     os.makedirs(output_dir, exist_ok=True)
#     logging.info(f"Analyzing clusters for dataset '{dataset_name}'")

#     # Generate cluster information
#     clusters_info = generate_cluster_info(
#         clustered_df,
#         text_column='short_description',
#         cluster_column='cluster',
#         sample_size=5,
#         output_dir=output_dir
#     )

#     # Initialize variables to avoid "not defined" errors
#     labeled_clusters = None
#     domains = None

#     # Label clusters using LLM
#     try:
#         labeled_clusters = label_clusters_with_llm(
#             clusters_info,
#             max_samples=300,
#             chunk_size=25,
#             output_dir=output_dir
#         )
#     except Exception as e:
#         logging.error(f"Error in cluster labeling: {e}")
#         logging.info("Creating basic labels as fallback")
#         # Create basic labels if LLM fails
#         labeled_clusters = {
#             str(cid): {"topic": f"Cluster {cid}", "description": "Auto-generated label"}
#             for cid in clustered_df['cluster'].unique() if cid != -1
#         }
#         labeled_clusters["-1"] = {"topic": "Noise", "description": "Unclustered data points"}

#         # Save fallback labels
#         with open(f"{output_dir}/labeled_clusters_fallback.json", "w") as f:
#             json.dump(labeled_clusters, f, indent=2, cls=NumpyEncoder)

#     # Group clusters into domains
#     try:
#         domains = group_clusters_into_domains(
#             labeled_clusters,
#             clusters_info,
#             num_domains=num_domains,
#             output_dir=output_dir
#         )
#     except Exception as e:
#         logging.error(f"Error in domain grouping: {e}")
#         logging.info("Creating basic domains as fallback")
#         # Create basic domains if grouping fails
#         domains = {"domains": [{"domain_name": "Other", "clusters": list(clustered_df['cluster'].unique())}]}

#         # Save fallback domains
#         with open(f"{output_dir}/domains_fallback.json", "w") as f:
#             json.dump(domains, f, indent=2, cls=NumpyEncoder)

#     # Apply labels to data
#     final_df = apply_labels_to_data(clustered_df, labeled_clusters, domains)

#     # Save final labeled dataframe - use chunking for large datasets
#     if len(final_df) > 100000:  # If the dataframe is very large
#         logging.info(f"Large dataset detected ({len(final_df)} rows). Saving in chunks...")
#         chunk_size = 50000
#         for i in range(0, len(final_df), chunk_size):
#             chunk = final_df.iloc[i:i+chunk_size]
#             if i == 0:
#                 # First chunk - create new file
#                 chunk.to_parquet(f"{output_dir}/final_df.parquet", index=False)
#             else:
#                 # Append subsequent chunks
#                 chunk.to_parquet(f"{output_dir}/final_df_part_{i//chunk_size}.parquet", index=False)

#         # Save a manifest file with all parts
#         with open(f"{output_dir}/final_df_manifest.json", "w") as f:
#             json.dump({
#                 "num_parts": (len(final_df) + chunk_size - 1) // chunk_size,
#                 "total_rows": len(final_df),
#                 "parts": [f"final_df.parquet"] +
#                         [f"final_df_part_{j}.parquet" for j in range(1, (len(final_df) + chunk_size - 1) // chunk_size)]
#             }, f, indent=2)
#     else:
#         # For smaller datasets, save as normal
#         final_df.to_parquet(f"{output_dir}/final_df.parquet", index=False)

#     # Log runtime
#     total_time = time.time() - start_time
#     logging.info(f"Cluster analysis completed in {total_time:.2f} seconds")

#     # Save runtime metadata
#     with open(f"{output_dir}/analysis_metadata.json", "w") as f:  # Correct syntax
#         metadata = {
#             "timestamp": datetime.now().isoformat(),
#             "parameters": {
#                 "num_domains": num_domains
#             },
#             "results": {
#                 "num_clusters": len([k for k in clusters_info.keys() if k != "-1"]),
#                 "num_domains": len(domains.get("domains", [])),
#                 "noise_percentage": clusters_info.get("-1", {}).get("percentage", 0) if "-1" in clusters_info else 0
#             },
#             "runtime_seconds": total_time,
#             "dataset_name": dataset_name
#         }
#         json.dump(metadata, f, indent=2, cls=NumpyEncoder)

#     return final_df, clusters_info, labeled_clusters, domains




def analyze_clusters_pipeline(clustered_df=None, dataset_name=None, num_domains=15, result_path="results"):
    """
    Third pipeline stage: Generate cluster info, label clusters and group into domains
    using the hybrid domain grouping approach with label standardization
    """
    start_time = time.time()

    # Check if we need to load data from file
    if clustered_df is None:
        if dataset_name is None:
            raise ValueError("Either clustered_df or dataset_name must be provided")

        # Load clustering results from file
        cluster_path = f"{result_path}/{dataset_name}/clustering/clustered_df.parquet"
        if not os.path.exists(cluster_path):
            raise FileNotFoundError(f"Clustered data not found at {cluster_path}")

        logging.info(f"Loading clustered data from {cluster_path} with memory optimization")
        try:
            # First try to load only essential columns for clustering
            essential_cols = ['number', 'short_description', 'cluster', 'cluster_probability', 'embedding']
            clustered_df = read_parquet_optimized(
                cluster_path,
                columns=essential_cols,
                chunksize=10  # Read 10 row groups at a time
            )
            logging.info(f"Loaded clustered data with {len(clustered_df)} rows and {len(clustered_df.columns)} columns")
        except Exception as e:
            logging.error(f"Error loading clustered data: {e}")
            raise

    # Create output directory
    output_dir = f"{result_path}/{dataset_name}/analysis"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Analyzing clusters for dataset '{dataset_name}'")

    # Generate cluster information
    clusters_info = generate_cluster_info(
        clustered_df,
        text_column='short_description',
        cluster_column='cluster',
        sample_size=5,
        output_dir=output_dir
    )

    # Initialize variables to avoid "not defined" errors
    labeled_clusters = None
    domains = None

    # Label clusters using LLM
    try:
        labeled_clusters = label_clusters_with_llm(
            clusters_info,
            max_samples=300,
            chunk_size=25,
            output_dir=output_dir
        )
    except Exception as e:
        logging.error(f"Error in cluster labeling: {e}")
        logging.info("Creating basic labels as fallback")
        # Create basic labels if LLM fails
        labeled_clusters = {
            str(cid): {"topic": f"Cluster {cid}", "description": "Auto-generated label"}
            for cid in clustered_df['cluster'].unique() if cid != -1
        }
        labeled_clusters["-1"] = {"topic": "Noise", "description": "Unclustered data points"}

        # Save fallback labels
        with open(f"{output_dir}/labeled_clusters_fallback.json", "w") as f:
            json.dump(labeled_clusters, f, indent=2, cls=NumpyEncoder)

    # Group clusters into domains using the hybrid approach

    # 3. Group clusters into domains
    try:
        # Load UMAP embeddings from file
        umap_embeddings_path = f"{result_path}/{dataset_name}/clustering/umap_embeddings.npy"
        if os.path.exists(umap_embeddings_path):
            umap_embeddings = np.load(umap_embeddings_path)

            # Load a small sample of the clustered data to get the cluster labels
            cluster_sample_path = f"{result_path}/{dataset_name}/clustering/clustered_df.parquet"
            sample_df = pd.read_parquet(cluster_sample_path, columns=['cluster'])
            cluster_labels = sample_df['cluster'].values

            # Call with all required parameters
            domains = group_clusters_into_domains(
                labeled_clusters,
                clusters_info,
                umap_embeddings,
                cluster_labels,
                # num_domains=15,
                max_domains=num_domains
                output_dir=f"{result_path}/{dataset_name}/analysis"
            )
        else:
            # Fallback if embeddings aren't available
            logging.warning("UMAP embeddings not found. Using fallback domain grouping.")
            domains = {"domains": [{"domain_name": "Other", "clusters": list(map(int, clusters_info.keys()))}]}
    except Exception as e:
        logging.error(f"Error in domain grouping: {e}")
        # Create basic domains if grouping fails
        domains = {"domains": [{"domain_name": "Other", "clusters": list(map(int, clusters_info.keys()))}]}
    # try:
    #     # Extract the umap_embeddings and cluster_labels from the clustered_df or load from file
    #     cluster_labels = clustered_df['cluster'].values

    #     # Try to load UMAP embeddings from file
    #     umap_embeddings_path = f"{result_path}/{dataset_name}/clustering/umap_embeddings.npy"
    #     if os.path.exists(umap_embeddings_path):
    #         logging.info(f"Loading UMAP embeddings from {umap_embeddings_path}")
    #         umap_embeddings = np.load(umap_embeddings_path)

    #         # Use the hybrid domain grouping approach
    #         domains = group_clusters_into_domains(
    #             labeled_clusters,
    #             clusters_info,
    #             umap_embeddings,
    #             cluster_labels,
    #             num_domains=num_domains,
    #             output_dir=output_dir
    #         )
    #     else:
    #         # If UMAP embeddings are not available, fall back to original method
    #         logging.warning("UMAP embeddings not found, falling back to original domain grouping method")
    #         domains = group_clusters_into_domains(
    #             labeled_clusters,
    #             clusters_info,
    #             umap_embeddings,
    #             cluster_labels,
    #             num_domains=num_domains,
    #             output_dir=output_dir,
    #             standardize_labels=False  # Don't standardize in the old method
    #         )
    # except Exception as e:
    #     logging.error(f"Error in domain grouping: {e}")
    #     logging.info("Creating basic domains as fallback")
    #     # Create basic domains if grouping fails
    #     domains = {"domains": [{"domain_name": "Other", "clusters": list(clustered_df['cluster'].unique())}]}

        # Save fallback domains
        with open(f"{output_dir}/domains_fallback.json", "w") as f:
            json.dump(domains, f, indent=2, cls=NumpyEncoder)

    # Apply labels to data
    final_df = apply_labels_to_data(clustered_df, labeled_clusters, domains)

    # Save final labeled dataframe - use chunking for large datasets
    if len(final_df) > 100000:  # If the dataframe is very large
        logging.info(f"Large dataset detected ({len(final_df)} rows). Saving in chunks...")
        chunk_size = 50000
        for i in range(0, len(final_df), chunk_size):
            chunk = final_df.iloc[i:i+chunk_size]
            if i == 0:
                # First chunk - create new file
                chunk.to_parquet(f"{output_dir}/final_df.parquet", index=False)
            else:
                # Append subsequent chunks
                chunk.to_parquet(f"{output_dir}/final_df_part_{i//chunk_size}.parquet", index=False)

        # Save a manifest file with all parts
        with open(f"{output_dir}/final_df_manifest.json", "w") as f:
            json.dump({
                "num_parts": (len(final_df) + chunk_size - 1) // chunk_size,
                "total_rows": len(final_df),
                "parts": [f"final_df.parquet"] +
                        [f"final_df_part_{j}.parquet" for j in range(1, (len(final_df) + chunk_size - 1) // chunk_size)]
            }, f, indent=2)
    else:
        # For smaller datasets, save as normal
        final_df.to_parquet(f"{output_dir}/final_df.parquet", index=False)

    # Log runtime
    total_time = time.time() - start_time
    logging.info(f"Cluster analysis completed in {total_time:.2f} seconds")

    # Save runtime metadata
    with open(f"{output_dir}/analysis_metadata.json", "w") as f:
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "num_domains": num_domains
            },
            "results": {
                "num_clusters": len([k for k in clusters_info.keys() if k != "-1"]),
                "num_domains": len(domains.get("domains", [])),
                "noise_percentage": clusters_info.get("-1", {}).get("percentage", 0) if "-1" in clusters_info else 0
            },
            "runtime_seconds": total_time,
            "dataset_name": dataset_name
        }
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)

    return final_df, clusters_info, labeled_clusters, domains

# %%
def save_results_pipeline(final_df=None, dataset_name=None, results_table_id=None,
                         write_disposition="WRITE_APPEND", result_path="results"):
    """
    Fourth pipeline stage: Save results to BigQuery
    """
    start_time = time.time()

    # Check if we need to load data from file
    if final_df is None:
        if dataset_name is None:
            raise ValueError("Either final_df or dataset_name must be provided")

        # Check if we have a manifest file (for large datasets)
        manifest_path = f"{result_path}/{dataset_name}/analysis/final_df_manifest.json"
        if os.path.exists(manifest_path):
            logging.info("Found manifest file for multi-part dataset")
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Load and combine all parts
            all_parts = []
            for part_file in manifest["parts"]:
                part_path = f"{result_path}/{dataset_name}/analysis/{part_file}"
                try:
                    logging.info(f"Loading part: {part_file}")
                    df_part = read_parquet_optimized(part_path)
                    all_parts.append(df_part)
                except Exception as e:
                    logging.error(f"Error loading part {part_file}: {e}")

            if all_parts:
                final_df = pd.concat(all_parts, ignore_index=True)
                logging.info(f"Combined {len(all_parts)} parts into dataframe with {len(final_df)} rows")
            else:
                raise ValueError("Could not load any parts of the final dataframe")
        else:
            # Regular single file
            final_path = f"{result_path}/{dataset_name}/analysis/final_df.parquet"
            if not os.path.exists(final_path):
                raise FileNotFoundError(f"Final data not found at {final_path}")

            logging.info(f"Loading final data from {final_path}")
            final_df = read_parquet_optimized(final_path)

    # Create output directory
    output_dir = f"{result_path}/{dataset_name}/final"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results for dataset '{dataset_name}'")

    # Save to BigQuery if table_id provided
    if results_table_id:
        try:
            # For very large dataframes, save to BigQuery in chunks
            if len(final_df) > 100000:
                chunk_size = 50000
                logging.info(f"Saving {len(final_df)} rows to BigQuery in chunks of {chunk_size}")

                for i in range(0, len(final_df), chunk_size):
                    chunk_df = final_df.iloc[i:i+chunk_size]
                    chunk_write_disposition = bigquery.WriteDisposition.WRITE_APPEND
                    if i == 0 and write_disposition == "WRITE_TRUNCATE":
                        chunk_write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

                    logging.info(f"Saving chunk {i//chunk_size + 1} with {len(chunk_df)} rows")
                    save_results_to_bigquery(chunk_df, results_table_id, str(chunk_write_disposition).split(".")[-1])
            else:
                save_results_to_bigquery(final_df, results_table_id, write_disposition)

            logging.info(f"Successfully saved results to BigQuery table {results_table_id}")
            success = True
        except Exception as e:
            logging.error(f"Error saving to BigQuery: {e}")
            success = False
    else:
        logging.info("No BigQuery table provided, skipping BigQuery save")
        success = False

    # Save final CSV locally for easy access - chunk if necessary
    if len(final_df) > 100000:
        chunk_size = 50000
        for i in range(0, len(final_df), chunk_size):
            chunk_df = final_df.iloc[i:i+chunk_size]
            if i == 0:
                chunk_df.to_csv(f"{output_dir}/results.csv", index=False)
            else:
                chunk_df.to_csv(f"{output_dir}/results_part_{i//chunk_size}.csv", index=False)
    else:
        final_df.to_csv(f"{output_dir}/results.csv", index=False)

    # Log runtime
    total_time = time.time() - start_time
    logging.info(f"Results saving completed in {total_time:.2f} seconds")

    return success

# %%
def run_modular_pipeline(
    input_query,
    embeddings_table_id,
    results_table_id,
    dataset_name,
    embedding_path=None,
    summary_path=None,
    write_disposition="WRITE_APPEND",
    min_cluster_size=25,
    min_samples=5,
    umap_n_components=50,
    num_domains=15,
    start_from_stage=1,
    end_at_stage=4,
    result_path="results",
    use_checkpoint=True  # Added parameter with default=True
):
    """
    Run the complete modular clustering pipeline, with ability to start/stop at specific stages

    Args:
        input_query: BigQuery query to load data
        embeddings_table_id: BigQuery table to save embeddings
        results_table_id: BigQuery table to save results
        dataset_name: Name for the dataset/run
        embedding_path: Optional custom path to load embeddings from
        summary_path: Optional path to precomputed summaries to skip LLM processing
        write_disposition: BigQuery write disposition
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        umap_n_components: Number of dimensions for UMAP reduction
        num_domains: Number of domains to group clusters into
        start_from_stage: Stage to start from (1-4)
        end_at_stage: Stage to end at (1-4)
        result_path: Base path to store results (default: "results")
        use_checkpoint: Whether to use checkpointing for UMAP embeddings (default: True)

    Returns:
        Dictionary with results from all completed stages
    """
    results = {}

    # Validate stages
    if not 1 <= start_from_stage <= 4:
        raise ValueError("start_from_stage must be between 1 and 4")
    if not 1 <= end_at_stage <= 4:
        raise ValueError("end_at_stage must be between 1 and 4")
    if start_from_stage > end_at_stage:
        raise ValueError("start_from_stage cannot be greater than end_at_stage")

    start_time = time.time()
    logging.info(f"Starting modular pipeline for dataset '{dataset_name}' (stages {start_from_stage}-{end_at_stage})")

    # Log embedding path if provided
    if embedding_path:
        logging.info(f"Using custom embedding path: {embedding_path}")
        # Validate the path exists if starting from stage 2 or later
        if start_from_stage > 1 and not os.path.exists(embedding_path):
            logging.warning(f"Warning: Custom embedding path {embedding_path} not found. Pipeline may fail.")

    # Stage 1: Generate embeddings
    if start_from_stage <= 1 <= end_at_stage:
        logging.info("=== STAGE 1: GENERATING EMBEDDINGS ===")
        df_with_embeddings, classification_result, fallback_stats = generate_embeddings_pipeline(
            input_query=input_query,
            dataset_name=dataset_name,
            embeddings_table_id=embeddings_table_id,
            write_disposition=write_disposition,
            summary_path=summary_path,  # Added parameter for precomputed summaries
            result_path=result_path
        )
        results["embeddings"] = {
            "df_with_embeddings": df_with_embeddings,
            "classification_result": classification_result,
            "fallback_stats": fallback_stats
        }

    # Stage 2: Train HDBSCAN
    if start_from_stage <= 2 <= end_at_stage:
        logging.info("=== STAGE 2: TRAINING HDBSCAN ===")
        # Use embeddings from stage 1 if available, otherwise use the provided path
        df_input = results.get("embeddings", {}).get("df_with_embeddings", None)

        clustered_df, umap_embeddings, clusterer, reducer = train_hdbscan_pipeline(
            df_with_embeddings=df_input,
            dataset_name=dataset_name,
            embedding_path=embedding_path,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            umap_n_components=umap_n_components,
            result_path=result_path,
            use_checkpoint=use_checkpoint  # Fixed parameter name (was user_checkpoint)
        )
        results["clustering"] = {
            "clustered_df": clustered_df,
            "umap_embeddings": umap_embeddings,
            "clusterer": clusterer,
            "reducer": reducer
        }

    # [rest of the function remains unchanged]
    # Stage 3: Analyze clusters
    if start_from_stage <= 3 <= end_at_stage:
        logging.info("=== STAGE 3: ANALYZING CLUSTERS ===")
        # Use clustering from stage 2 if available
        df_input = results.get("clustering", {}).get("clustered_df", None)

        final_df, clusters_info, labeled_clusters, domains = analyze_clusters_pipeline(
            clustered_df=df_input,
            dataset_name=dataset_name,
            num_domains=num_domains,
            result_path=result_path
        )
        results["analysis"] = {
            "final_df": final_df,
            "clusters": clusters_info,
            "labeled_clusters": labeled_clusters,
            "domains": domains
        }

    # Stage 4: Save results
    if start_from_stage <= 4 <= end_at_stage:
        logging.info("=== STAGE 4: SAVING RESULTS ===")
        # Use final dataframe from stage 3 if available
        df_input = results.get("analysis", {}).get("final_df", None)

        success = save_results_pipeline(
            final_df=df_input,
            dataset_name=dataset_name,
            results_table_id=results_table_id,
            write_disposition=write_disposition,
            result_path=result_path
        )
        results["saved"] = success

    # Calculate total runtime
    total_time = time.time() - start_time
    logging.info(f"Modular pipeline completed in {total_time:.2f} seconds")

    # Create consolidated metadata
    output_dir = f"{result_path}/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/run_metadata.json", "w") as f:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "stages_run": list(range(start_from_stage, end_at_stage + 1)),
            "parameters": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "umap_n_components": umap_n_components,
                "num_domains": num_domains,
                "write_disposition": write_disposition,
                "use_checkpoint": use_checkpoint  # Added to metadata
            },
            "runtime_seconds": total_time
        }
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)

    return results

# %%


# %%


# %%
# Modified save_run_metadata function
def save_run_metadata(results, run_params, dataset_name, output_dir=None, result_path="results"):
    """Save consolidated metadata about the clustering run"""

    # Use provided output_dir or construct from dataset_name
    output_dir = output_dir or f"{result_path}/{dataset_name}"

    # Create run metadata
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "parameters": run_params,
        "statistics": {
            "num_records": len(results["clustered_df"]),
            "num_clusters": len([k for k in results["clusters"] if k != "-1"]),
            "num_domains": len(results["domains"]["domains"]),
            "noise_percentage": results["clusters"].get("-1", {}).get("percentage", 0) if "-1" in results["clusters"] else 0,
            "runtime_seconds": results["runtime_seconds"]
        },
        "artifacts": {
            "cluster_details": "cluster_details.json",
            "labeled_clusters": "labeled_clusters.json",
            "domains": "domains.json",
            "umap_reducer": "umap_reducer.pkl",
            "hdbscan_clusterer": "hdbscan_clusterer.pkl"
        }
    }

    # Save metadata
    with open(f"{output_dir}/run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved run metadata to {output_dir}/run_metadata.json")
    return run_id

# %%
def evaluate_clustering_quality(df, cluster_column='cluster', embeddings_column='embedding', result_path="results", dataset_name=None):
    """
    Evaluate the quality of clustering using metrics like silhouette score and within-cluster similarity

    Args:
        df: DataFrame with clustering results
        cluster_column: Column containing cluster labels
        embeddings_column: Column containing embeddings as JSON strings
        result_path: Base path for results
        dataset_name: Name of the dataset

    Returns:
        Dictionary with evaluation metrics
    """
    logging.info("Evaluating clustering quality...")

    # Extract embeddings from the dataframe
    embeddings = np.array([json.loads(emb) for emb in df[embeddings_column].tolist()])

    # Get cluster labels
    cluster_labels = df[cluster_column].values

    # Calculate metrics only on non-noise points
    non_noise_mask = cluster_labels != -1

    if np.sum(non_noise_mask) <= 1:
        logging.warning("Too few non-noise points to calculate metrics")
        return {"error": "Too few non-noise points"}

    try:
        # Calculate silhouette score if at least 2 clusters
        unique_clusters = set(cluster_labels[non_noise_mask])
        metrics = {}

        if len(unique_clusters) > 1:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(
                embeddings[non_noise_mask],
                cluster_labels[non_noise_mask],
                metric='cosine',
                sample_size=min(10000, np.sum(non_noise_mask))  # Limit sample size for large datasets
            )
            metrics["silhouette_score"] = float(silhouette)

        # Calculate cluster statistics
        unique_labels = np.unique(cluster_labels)
        cluster_stats = {}

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            cluster_mask = cluster_labels == label
            cluster_size = np.sum(cluster_mask)

            cluster_stats[int(label)] = {
                "size": int(cluster_size),
                "percentage": float(round((cluster_size / len(df)) * 100, 2))
            }

        metrics["cluster_stats"] = cluster_stats
        metrics["num_clusters"] = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        metrics["noise_percentage"] = float(round((np.sum(cluster_labels == -1) / len(df)) * 100, 2))

        # Save metrics
        if dataset_name:
            output_dir = f"{result_path}/{dataset_name}/evaluation"
            os.makedirs(output_dir, exist_ok=True)

            with open(f"{output_dir}/clustering_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, cls=NumpyEncoder)

        return metrics

    except Exception as e:
        logging.error(f"Error calculating clustering metrics: {e}")
        return {"error": str(e)}

# %%
def run_clustering_pipeline(
    input_query,
    embeddings_table_id,
    results_table_id,
    dataset_name,  # Add this parameter
    write_disposition="WRITE_APPEND",
    min_cluster_size=25,
    min_samples=5,
    umap_n_components=50,
    num_domains=15,
    result_path="results"
):
    """Run the complete clustering pipeline"""

    start_time = time.time()
    logging.info("Starting clustering pipeline")

    # Create output directory with dataset name
    output_dir = f"{result_path}/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results to {output_dir}")

    # 1. Load data
    logging.info(f"Loading data with query: {input_query}")
    df = run_query(input_query)
    logging.info(f"Loaded {len(df)} records")

    # 2. Generate hybrid embeddings
    df_with_embeddings, classification_result, fallback_stats = create_hybrid_embeddings(
    df, batch_size=25
)

    # 3. Save embeddings to BigQuery
    save_embeddings_to_bigquery(df_with_embeddings, embeddings_table_id)

    # 4. Run HDBSCAN clustering
    clustered_df, umap_embeddings, clusterer, reducer = hybrid_embedding_hdbscan_clustering(
        df_with_embeddings,
        embedding_col='embedding',
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        umap_n_components=umap_n_components
    )
    cluster_labels = clustered_df['cluster'].values

    # Save model artifacts locally (updated paths)
    with open(f"{output_dir}/umap_reducer.pkl", "wb") as f:
        pickle.dump(reducer, f)

    with open(f"{output_dir}/hdbscan_clusterer.pkl", "wb") as f:
        pickle.dump(clusterer, f)

    # 5. Generate cluster information
    clusters = generate_cluster_info(
        clustered_df, text_column='short_description', cluster_column='cluster',
        output_dir=output_dir  # Pass output directory
    )

    # 6. Label clusters using OpenAI
    labeled_clusters = label_clusters_with_llm(clusters, output_dir=output_dir)

    # 7. Group clusters into domains
    domains = group_clusters_into_domains(
        labeled_clusters,
        clusters,
        umap_embeddings,
        cluster_labels,
        max_domains=num_domains,  # Changed from num_domains=num_domains
        output_dir=output_dir
    )
    # 8. Apply labels to data
    final_df = apply_labels_to_data(clustered_df, labeled_clusters, domains)

    # 9. Save results to BigQuery
    save_results_to_bigquery(final_df, results_table_id, write_disposition)

    # Calculate and log total runtime
    total_time = time.time() - start_time
    logging.info(f"Pipeline completed in {total_time:.2f} seconds")

    run_params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "umap_n_components": umap_n_components,
        "num_domains": num_domains,
        "dataset_name": dataset_name  # Include dataset name in params
    }

    # Save run metadata with dataset name
    save_run_metadata({
        "clustered_df": final_df,
        "clusters": clusters,
        "labeled_clusters": labeled_clusters,
        "domains": domains,
        "runtime_seconds": total_time
    }, run_params, dataset_name, result_path=result_path)

    # Return results
    return {
        "clustered_df": final_df,
        "clusters": clusters,
        "labeled_clusters": labeled_clusters,
        "domains": domains,
        "umap_embeddings": umap_embeddings,
        "runtime_seconds": total_time
    }

# %%


# %%
def save_minimal_results_to_bigquery(df, minimal_table_id):
    """
    Save minimal results to BigQuery (just number, category and subcategory)
    """
    logging.info(f"Saving {len(df)} minimal results to {minimal_table_id}")

    # Create minimal dataframe with just the essential columns
    minimal_df = df[['number', 'combined_incidents_summary','category', 'subcategory']].copy()

    # Connect to BigQuery
    client, _ = connect_to_bq()
    sanitized_table_id = minimal_table_id.replace('`', '')

    # Configure job
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("number", "STRING"),
            bigquery.SchemaField("combined_incidents_summary", "STRING"),
            bigquery.SchemaField("category", "STRING"),
            bigquery.SchemaField("subcategory", "STRING")
        ],
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )

    # Execute load job
    try:
        job = client.load_table_from_dataframe(
            minimal_df,
            sanitized_table_id,
            job_config=job_config,
            timeout=300  # 5 minutes timeout
        )
        job.result()  # Wait for job to complete
        logging.info(f"Successfully saved {len(minimal_df)} minimal results to {minimal_table_id}")
    except Exception as e:
        logging.error(f"Error saving minimal results: {e}")
        raise

# %%
# Set the dataset name for organizing results
dataset_name = "pde_2024_sample"  # Change this to your desired dataset name
result_path = r"D:\CollectGuestLogsTemp\results"    # Base path for all saved outputs - can be modified to any directory
embedding_path = r"D:\CollectGuestLogsTemp\results"  # Your custom path

# Create results directory structure with dataset name
os.makedirs(f"{result_path}/{dataset_name}", exist_ok=True)
os.makedirs(embedding_path, exist_ok=True)

# %%
# Define your query and table IDs
tech_center_query = """
SELECT t3.number, t3.priority, t3.sys_created_on, t1.TeamDepartment, t1.Area, t1.Unit, t1.TechCenter,
       t3.short_description, t3.close_code, t3.vendor, t3.assignment_group, t3.state, t3.business_service,
       t3.close_notes, t3.work_notes, t3.contact_type as chanel, t3.description
FROM `enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev` t3
JOIN `enterprise-dashboardnp-cd35.bigquery_datasets_spoke_oa_dev.Team_services` t1
ON t3.business_service = t1.Services
WHERE t1.TechCenter = 'BT-TC-Product Development & Engineering'
AND t3.sys_created_on >= '2024-01-01' AND t3.sys_created_on < '2025-01-01'
"""

# %%
# Define your table IDs
embeddings_table_id = "`enterprise-dashboardnp-cd35.bigquery_datasets_spoke_oa_dev.Incidents_embeddings_hybrid_hdbscan_pde_output_2024_4o_mini_bach`"
results_table_id = "`enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev_hybrid_hdbscan_pde_output_2024_4o_mini_bach`"



# embeddings_table_id = "`enterprise-dashboardnp-cd35.bigquery_datasets_spoke_oa_dev.Incidents_embeddings_hybrid_hdbscan_pde_output_2024_bach_sample`"
# results_table_id = "`enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev_hybrid_hdbscan_pde_output_2024_bach_sample`"




# %% [markdown]
# ###### 1. Generating Embeddings: Creates vector representations of incident text using Azure OpenAI
# ###### 2. Training HDBSCAN: Applies dimensionality reduction (UMAP) and clustering
# ###### 3. Analyzing Clusters: Generates cluster information and labels using LLM
# ###### 4. Saving Results: Saves the final labeled data to BigQuery

# %%
start_from_stage = 2 #efer to the above

# %%
# Free up memory before running
import gc
gc.collect()

# Set memory optimization parameters
import os
os.environ["PYARROW_BUFFER_MAX_ALLOCATION"] = str(1 * 1024 * 1024 * 1024)  # 1GB max allocation

try:
    results = run_modular_pipeline(
                input_query=tech_center_query,
                embeddings_table_id=embeddings_table_id,
                results_table_id=results_table_id,
                dataset_name=dataset_name,
                embedding_path=f"{result_path}/{dataset_name}/embeddings/df_with_embeddings.parquet",
                write_disposition="WRITE_TRUNCATE",
                min_cluster_size=12,      # Try 10–15 for more, smaller clusters
                min_samples=5,
                umap_n_components=150,    # Try 100–200 for more detail
                num_domains=20,           # Try 25–30 for more domain separation
                start_from_stage=start_from_stage,
                end_at_stage=4,
                result_path=result_path,
                use_checkpoint=True# Controls whether to use UMAP checkpointing
)

    print(f"Pipeline completed for dataset: {dataset_name}")
    if "analysis" in results:
        print(f"Found {len(results['analysis']['labeled_clusters'])} labeled clusters")
        print(f"Organized into {len(results['analysis']['domains']['domains'])} domains")
        print(f"Processed {len(results['analysis']['final_df'])} records")

except Exception as e:
    # If we still get memory errors, try a more radical approach
    import logging
    logging.error(f"Error running pipeline: {e}")
    print("Trying alternative approach with minimal memory usage...")

    # Alternative approach: Run each stage separately with manual garbage collection
    try:
        # 1. Skip to loading cluster details that are already saved
        cluster_details_path = f"{result_path}/{dataset_name}/clustering/cluster_details.json"
        if not os.path.exists(cluster_details_path):
            # Generate new cluster details by loading only cluster and text cols
            cluster_path = f"{result_path}/{dataset_name}/clustering/clustered_df.parquet"
            cols_needed = ['cluster', 'short_description']
            clustered_df_minimal = pd.read_parquet(cluster_path, columns=cols_needed)

            # Create output directory for analysis
            output_dir = f"{result_path}/{dataset_name}/analysis"
            os.makedirs(output_dir, exist_ok=True)

            # Generate cluster info with minimal data
            clusters_info = generate_cluster_info(
                clustered_df_minimal,
                text_column='short_description',
                cluster_column='cluster',
                sample_size=5,
                output_dir=output_dir
            )
            del clustered_df_minimal
            gc.collect()
        else:
            # Load existing cluster details
            with open(cluster_details_path, "r") as f:
                clusters_info = json.load(f)

        # 2. Label clusters with LLM
        labeled_clusters = label_clusters_with_llm(
            clusters_info,
            max_samples=300,
            chunk_size=25,
            output_dir=f"{result_path}/{dataset_name}/analysis"
        )

        # 3. Group clusters into domains
        try:
            # Load UMAP embeddings from file
            umap_embeddings_path = f"{result_path}/{dataset_name}/clustering/umap_embeddings.npy"
            umap_embeddings = np.load(umap_embeddings_path)

            # Load or extract cluster labels
            if 'clustered_df_minimal' in locals():
                cluster_labels = clustered_df_minimal['cluster'].values
            else:
                # If clustered_df_minimal was already deleted, reload minimal data
                cluster_path = f"{result_path}/{dataset_name}/clustering/clustered_df.parquet"
                cluster_df = pd.read_parquet(cluster_path, columns=['cluster'])
                cluster_labels = cluster_df['cluster'].values
                del cluster_df
                gc.collect()

            # Fixed parameter name: num_domains → max_domains
            domains = group_clusters_into_domains(
                labeled_clusters,
                clusters_info,
                umap_embeddings,
                cluster_labels,
                max_domains=15,  # Changed from num_domains=15
                output_dir=f"{result_path}/{dataset_name}/analysis"
            )
        except Exception as e:
            logging.error(f"Error in domain grouping: {e}")
            # Fallback to a simpler domain structure
            domains = {"domains": [
                {"domain_name": "Other", "description": "All clusters",
                "clusters": [int(k) for k in clusters_info.keys() if k != "-1"]}
            ]}
            domains["domains"].append({"domain_name": "Noise", "description": "Uncategorized incidents", "clusters": [-1]})

        # Save fallback domains
        with open(f"{result_path}/{dataset_name}/analysis/domains_fallback.json", "w") as f:
            json.dump(domains, f, indent=2, cls=NumpyEncoder)

        # 4. Apply labels to clustered data
        cluster_path = f"{result_path}/{dataset_name}/clustering/clustered_df.parquet"

        # Process in chunks to avoid memory issues - improved approach
        chunk_size = 10000
        parquet_file = pq.ParquetFile(cluster_path)
        total_rows = parquet_file.metadata.num_rows

        # Process and save in chunks with more robust reading approach
        output_dir = f"{result_path}/{dataset_name}/analysis"
        for i in range(0, total_rows, chunk_size):
            # Try direct slicing approach which is more reliable
            end_idx = min(i + chunk_size, total_rows)
            df_chunk = pd.read_parquet(
                cluster_path,
                engine="pyarrow",
            ).iloc[i:i+chunk_size]

            # Apply labels
            df_chunk = apply_labels_to_data(df_chunk, labeled_clusters, domains)

            # Save chunk
            if i == 0:
                df_chunk.to_parquet(f"{output_dir}/final_df.parquet", index=False)
            else:
                df_chunk.to_parquet(f"{output_dir}/final_df_part_{i//chunk_size}.parquet", index=False)

            # Save manifest after each chunk to track progress
            with open(f"{output_dir}/final_df_manifest.json", "w") as f:
                json.dump({
                    "processed_rows": min(i + chunk_size, total_rows),
                    "total_rows": total_rows,
                    "num_parts": (i // chunk_size) + 1,
                    "parts": [f"final_df.parquet"] +
                            [f"final_df_part_{j}.parquet" for j in range(1, (i // chunk_size) + 1)]
                }, f, indent=2)

            # Force garbage collection
            del df_chunk
            gc.collect()

        print("Chunked processing complete. Final step: save to BigQuery")

        # 5. Save to BigQuery in chunks
        success = save_results_pipeline(
            final_df=None,  # Will load from files
            dataset_name=dataset_name,
            results_table_id=results_table_id,
            write_disposition="WRITE_TRUNCATE",
            result_path=result_path
        )

        print(f"Pipeline completed with chunked approach: {success}")

    except Exception as detailed_error:
        logging.error(f"Even alternative approach failed: {detailed_error}")
        print(f"All approaches failed. Please try with smaller data or more memory: {detailed_error}")

# %%
# Access clustering analysis results

# Method 1: If you have the results from the pipeline in memory
if 'results' in globals() and 'analysis' in results:
    print("Method 1: Accessing results from pipeline output")
    clusters_info = results['analysis']['clusters']
    labeled_clusters = results['analysis']['labeled_clusters']
    domains = results['analysis']['domains']
    final_df = results['analysis']['final_df']

# Method 2: Load results from files (if not in memory or running separately)
else:
    print("Method 2: Loading results from saved files")
    # Define paths
    analysis_dir = f"{result_path}/{dataset_name}/analysis"

    # Load cluster details
    with open(f"{analysis_dir}/cluster_details.json", "r") as f:
        clusters_info = json.load(f)

    # Load labeled clusters
    with open(f"{analysis_dir}/labeled_clusters.json", "r") as f:
        labeled_clusters = json.load(f)

    # Load domain groupings
    with open(f"{analysis_dir}/domains.json", "r") as f:
        domains = json.load(f)

    # Load final dataframe with labels
    final_df = read_parquet_optimized(f"{analysis_dir}/final_df.parquet")

# Print basic statistics
print(f"\n===== CLUSTERING STATISTICS =====")
n_clusters = len([k for k in clusters_info.keys() if k != "-1"])
n_noise = clusters_info.get("-1", {}).get("size", 0) if "-1" in clusters_info else 0
total = sum(clusters_info[k]["size"] for k in clusters_info)
print(f"Total records: {total}")
print(f"Number of clusters: {n_clusters}")
print(f"Noise points: {n_noise} ({clusters_info.get('-1', {}).get('percentage', 0)}% of data)")
print(f"Number of domains: {len(domains['domains'])}")

# Print top 10 largest clusters with their topics
print("\n===== TOP 10 LARGEST CLUSTERS =====")
top_clusters = sorted(
    [c for c in clusters_info.keys() if c != "-1"],
    key=lambda c: clusters_info[c]["size"],
    reverse=True
)[:10]

for cluster_id in top_clusters:
    cluster = clusters_info[cluster_id]
    # Fix: Ensure consistent type handling for cluster IDs
    if cluster_id in labeled_clusters:
        topic = labeled_clusters[cluster_id]["topic"]
    elif str(cluster_id) in labeled_clusters:  # Try as string
        topic = labeled_clusters[str(cluster_id)]["topic"]
    else:  # Fallback if not found
        topic = f"Unlabeled Cluster {cluster_id}"
    print(f"Cluster {cluster_id}: {topic} - {cluster['size']} incidents ({cluster['percentage']}%)")

# Print domains with their clusters
print("\n===== DOMAINS AND THEIR CLUSTERS =====")
for i, domain in enumerate(domains["domains"]):
    if domain["domain_name"] == "Noise":
        continue
    print(f"{i+1}. {domain['domain_name']}: {domain['description']}")
    domain_clusters = domain["clusters"]
    for cluster_id in domain_clusters:
        if str(cluster_id) in labeled_clusters:
            topic = labeled_clusters[str(cluster_id)]["topic"]
            size = clusters_info[str(cluster_id)]["size"]
            print(f"   - Cluster {cluster_id}: {topic} ({size} incidents)")
    print()

# Create a visualization of cluster sizes (excluding noise)
plt.figure(figsize=(14, 8))
cluster_ids = [c for c in clusters_info.keys() if c != "-1"]
sizes = [clusters_info[c]["size"] for c in cluster_ids]
labels = [labeled_clusters[c]["topic"] for c in cluster_ids]

# Sort by size for better visualization
sorted_indices = np.argsort(sizes)[::-1]  # Descending order
sorted_sizes = [sizes[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
sorted_ids = [cluster_ids[i] for i in sorted_indices]

# Plot only top 20 for readability
plt.bar(range(min(20, len(sorted_sizes))), sorted_sizes[:20], color='steelblue')
plt.xticks(range(min(20, len(sorted_sizes))), [f"{id}: {label[:15]}..." for id, label in zip(sorted_ids[:20], sorted_labels[:20])], rotation=90)
plt.title('Top 20 Clusters by Size')
plt.xlabel('Cluster ID and Topic')
plt.ylabel('Number of Incidents')
plt.tight_layout()
plt.show()

# Sample records from a specific cluster (choose a large one)
if len(top_clusters) > 0:
    sample_cluster_id = top_clusters[0]
    print(f"\n===== SAMPLE RECORDS FROM CLUSTER {sample_cluster_id} =====")
    print(f"Topic: {labeled_clusters[sample_cluster_id]['topic']}")
    print(f"Description: {labeled_clusters[sample_cluster_id]['description']}")

    cluster_samples = final_df[final_df['cluster'] == int(sample_cluster_id)].head(5)
    for _, row in cluster_samples.iterrows():
        print(f"- {row['short_description']}")

# %%
test_df = run_query("""select * from `enterprise-dashboardnp-cd35.bigquery_datasets_spoke_oa_dev.Incidents_embeddings_hybrid_hdbscan_pde_output_2024_4o_mini_bach` limit 100""")

# %%
embedding_df = run_query("""select * from `enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev_hybrid_hdbscan_pde_output_2024_4o_mini_bach`
""")

# %%
results_df.head()

# %%
unlabled = results_df[results_df['cluster'] == 169]

# %%
query = """select distinct business_service from  `enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev` """

# %%
tech_list = run_query(query)

# %%
for i in tech_list['business_service']:
    print(i)

# %%



