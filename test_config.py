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
            
        # Test config access
        print(f"   BigQuery Project: {config.bigquery.project_id}")
        print(f"   Tech Centers: {len(config.tech_centers)}")
        print(f"   Embedding Weights: {config.clustering.embedding_weights}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
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

def main():
    """Run all tests"""
    print("🧪 Running configuration and component tests...")
    print("=" * 60)
    
    tests = [
        test_environment_variables,
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