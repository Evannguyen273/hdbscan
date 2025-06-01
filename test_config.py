#!/usr/bin/env python3
"""
Test script to verify configuration loading and basic functionality
Run this to ensure everything is working after the config refactoring
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing configuration loading...")
    
    try:
        from config.config import load_config, get_config, validate_environment
        
        # Test loading config
        config = load_config()
        print("✅ Configuration loaded successfully")
        
        # Test environment validation  
        if validate_environment():
            print("✅ Environment validation passed")
        else:
            print("❌ Environment validation failed")
            return False
            
        # Test config access with detailed credential checking
        print(f"   BigQuery Project: {config.bigquery.project_id}")
        print(f"   Tech Centers: {len(config.tech_centers)}")
        
        # Test service account credential loading
        service_account = config.bigquery.service_account_key_path
        if isinstance(service_account, dict):
            print("✅ Service account credentials loaded as JSON object")
            print(f"   Project ID from credentials: {service_account.get('project_id', 'N/A')}")
        else:
            print("❌ Service account credentials not properly loaded")
            return False
            
        # Test table configurations
        tables = config.bigquery.tables
        print(f"   Incident table: {tables.get('incidents', 'N/A')}")
        print(f"   Team services table: {tables.get('team_services', 'N/A')}")
        
        # Test Azure OpenAI configuration
        azure_config = config.azure.openai
        print(f"   Azure OpenAI endpoint: {azure_config.get('endpoint', 'N/A')[:30]}...")
        print(f"   Embedding model: {azure_config.get('embedding_model', 'N/A')}")
        
        # Test blob storage configuration
        blob_config = config.get('blob_storage', {})
        blob_conn = blob_config.get('connection_string', '')
        if blob_conn:
            print(f"   Blob storage: {blob_conn[:30]}...")
        else:
            print("❌ Blob storage connection not configured")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bigquery_client():
    """Test BigQuery client initialization"""
    print("\n🗄️ Testing BigQuery client...")
    
    try:
        from config.config import get_config
        from data.bigquery_client import BigQueryClient
        
        config = get_config()
        bq_client = BigQueryClient(config.config)
        
        print("✅ BigQuery client initialized successfully")
        print(f"   Project ID: {bq_client.project_id}")
        print(f"   Tables configured: {len(bq_client.tables)}")
        
        # Test table access
        incident_table = bq_client.get_table_id('raw_incidents')
        print(f"   Incident table: {incident_table}")
        
        return True
        
    except Exception as e:
        print(f"❌ BigQuery client test failed: {e}")
        return False

def test_embedding_generator():
    """Test embedding generator initialization"""
    print("\n🧠 Testing embedding generator...")
    
    try:
        from config.config import get_config
        from preprocessing.embedding_generation import EmbeddingGenerator
        
        config = get_config()
        embedding_gen = EmbeddingGenerator(config.config)
        
        print("✅ Embedding generator initialized successfully")
        print(f"   Model: {embedding_gen.embedding_model}")
        print(f"   Weights: {embedding_gen.embedding_weights}")
        print(f"   Batch size: {embedding_gen.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding generator test failed: {e}")
        return False

def test_blob_storage():
    """Test blob storage client initialization"""
    print("\n☁️ Testing blob storage client...")
    
    try:
        from config.config import get_config
        from utils.blob_storage import BlobStorageClient
        
        config = get_config()
        blob_client = BlobStorageClient(config.config)
        
        print("✅ Blob storage client initialized successfully")
        print(f"   Container: {blob_client.container_name}")
        print(f"   Connection: {blob_client.connection_string[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Blob storage test failed: {e}")
        return False

def test_environment_variables():
    """Test that all required environment variables are present"""
    print("\n🌍 Testing environment variables...")
    
    required_vars = [
        'SERVICE_ACCOUNT_KEY_PATH',
        'TEAM_SERVICES_TABLE', 
        'INCIDENT_TABLE',
        'PROBLEM_TABLE',
        'AZURE_OPENAI_ENDPOINT',
        'OPENAI_API_KEY',
        'AZURE_OPENAI_EMBEDDING_ENDPOINT',
        'BLOB_CONNECTION_STRING'
    ]
      missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            # Mask sensitive values for display
            value = os.getenv(var)
            if len(value) > 20:
                masked_value = value[:10] + "..." + value[-10:]
            else:
                masked_value = "*" * len(value)
            print(f"   ✅ {var}: {masked_value}")
    
    if missing_vars:
        print(f"   ❌ Missing variables: {missing_vars}")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def test_credential_parsing():
    """Test specific credential parsing"""
    print("\n🔑 Testing credential parsing...")
    
    try:
        import json
        
        # Test service account JSON parsing
        service_account_json = os.getenv('SERVICE_ACCOUNT_KEY_PATH')
        if service_account_json:
            try:
                parsed_creds = json.loads(service_account_json)
                print("✅ Service account JSON parsed successfully")
                print(f"   Type: {parsed_creds.get('type', 'N/A')}")
                print(f"   Project ID: {parsed_creds.get('project_id', 'N/A')}")
                print(f"   Client email: {parsed_creds.get('client_email', 'N/A')[:20]}...")
                
                # Check required fields
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                missing_fields = [field for field in required_fields if field not in parsed_creds]
                if missing_fields:
                    print(f"   ❌ Missing required fields: {missing_fields}")
                    return False
                else:
                    print("✅ All required service account fields present")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse service account JSON: {e}")
                return False
        else:
            print("❌ SERVICE_ACCOUNT_KEY_PATH not found")
            return False
            
        # Test table name formats
        tables = {
            'TEAM_SERVICES_TABLE': os.getenv('TEAM_SERVICES_TABLE'),
            'INCIDENT_TABLE': os.getenv('INCIDENT_TABLE'),
            'PROBLEM_TABLE': os.getenv('PROBLEM_TABLE')
        }
        
        for table_name, table_id in tables.items():
            if table_id and '.' in table_id:
                parts = table_id.split('.')
                if len(parts) == 3:
                    print(f"✅ {table_name}: Valid format (project.dataset.table)")
                else:
                    print(f"❌ {table_name}: Invalid format - should be project.dataset.table")
                    return False
            else:
                print(f"❌ {table_name}: Missing or invalid format")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Credential parsing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running configuration and component tests...")
    print("=" * 60)
      tests = [
        test_environment_variables,
        test_credential_parsing,
        test_configuration,
        test_bigquery_client,
        test_embedding_generator,
        test_blob_storage
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Your configuration is working correctly.")
        print("\n🚀 You can now run:")
        print("   python main.py validate")
        print("   python main.py status")
        print("   python main.py preprocess --limit 10")
    else:
        print("⚠️ Some tests failed. Please check your .env file and configuration.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)