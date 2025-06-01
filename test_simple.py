#!/usr/bin/env python3
"""
Simple script to test OpenAI and Azure credentials
"""

import os
from dotenv import load_dotenv

def test_openai_credentials():
    """Test OpenAI API credentials"""
    print("🔑 Testing OpenAI Credentials...")
    print("=" * 40)
    
    # Load .env file
    load_dotenv()
    
    # Test OpenAI Chat
    print("\n💬 Testing OpenAI Chat:")
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('OPENAI_API_KEY')
    deployment = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')
    
    if endpoint and api_key and deployment:
        print(f"   ✅ Endpoint: {endpoint[:30]}...")
        print(f"   ✅ API Key: {'*' * 20}")
        print(f"   ✅ Deployment: {deployment}")
        
        # Try to make a simple API call
        try:
            import openai
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2023-12-01-preview"
            )
            
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "Hello, test message"}],
                max_tokens=10
            )
            print("   ✅ Chat API test successful!")
            
        except Exception as e:
            print(f"   ❌ Chat API test failed: {e}")
    else:
        print("   ❌ Missing chat credentials")
    
    # Test OpenAI Embeddings
    print("\n🧠 Testing OpenAI Embeddings:")
    embed_endpoint = os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT')
    embed_key = os.getenv('AZURE_OPENAI_EMBEDDING_KEY')
    embed_model = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL')
    
    if embed_endpoint and embed_key and embed_model:
        print(f"   ✅ Endpoint: {embed_endpoint[:30]}...")
        print(f"   ✅ API Key: {'*' * 20}")
        print(f"   ✅ Model: {embed_model}")
        
        # Try to make a simple embedding call
        try:
            import requests
            
            headers = {
                'Content-Type': 'application/json',
                'api-key': embed_key
            }
            
            data = {
                'input': 'test embedding',
                'model': embed_model
            }
            
            response = requests.post(embed_endpoint, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                print("   ✅ Embedding API test successful!")
            else:
                print(f"   ❌ Embedding API test failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Embedding API test failed: {e}")
    else:
        print("   ❌ Missing embedding credentials")
    
    # Test Blob Storage
    print("\n☁️ Testing Blob Storage:")
    blob_conn = os.getenv('BLOB_CONNECTION_STRING')
    
    if blob_conn:
        print(f"   ✅ Connection: {blob_conn[:30]}...")
        
        try:
            from azure.storage.blob import BlobServiceClient
            
            blob_client = BlobServiceClient.from_connection_string(blob_conn)
            
            # Try to list containers (simple test)
            containers = list(blob_client.list_containers(max_results=1))
            print("   ✅ Blob storage connection successful!")
            
        except Exception as e:
            print(f"   ❌ Blob storage test failed: {e}")
    else:
        print("   ❌ Missing blob storage connection")
    
    print("\n" + "=" * 40)
    print("🏁 Credential testing complete!")

if __name__ == "__main__":
    test_openai_credentials()