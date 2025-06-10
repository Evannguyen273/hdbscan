#!/usr/bin/env python3
"""
Test script to verify Azure OpenAI chat and embedding functionality with configuration
"""
import sys
import os
import logging
from pathlib import Path
from openai import AzureOpenAI
import numpy as np

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_azure_openai_chat():
    """Test Azure OpenAI Chat using configuration"""
    print("\n" + "=" * 60)
    print("🤖 Testing Azure OpenAI Chat")
    print("=" * 60)

    try:
        # Import config
        from config.config import get_config

        # Get configuration
        config = get_config()

        # Access Azure OpenAI configuration via get()
        endpoint = config.get('azure.openai.endpoint')
        api_key = config.get('azure.openai.api_key')
        api_version = config.get('azure.openai.api_version')
        deployment_name = config.get('azure.openai.deployment_name')

        # Display configuration
        print(f"Using endpoint: {endpoint[:30]}..." if endpoint else "Endpoint not configured")
        print(f"Using deployment: {deployment_name}" if deployment_name else "Deployment not configured")
        print(f"Using API version: {api_version}" if api_version else "API version not configured")

        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )

        # Simple chat message
        print("\n📤 Sending message: 'Hi, how are you?'")

        # Get chat completion
        completion = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi, how are you?"}
            ],
            max_tokens=100
        )

        # Display response
        response = completion.choices[0].message.content
        print(f"\n📥 Response: \n{response}")

        print("\n✅ Azure OpenAI Chat test successful!")
        return True

    except Exception as e:
        print(f"\n❌ Azure OpenAI Chat test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_azure_openai_embeddings():
    """Test Azure OpenAI Embedding using configuration"""
    print("\n" + "=" * 60)
    print("🧠 Testing Azure OpenAI Embeddings")
    print("=" * 60)

    try:
        # Import config
        from config.config import get_config

        # Get configuration
        config = get_config()

        # Access Azure OpenAI embedding configuration via get()
        embedding_endpoint = config.get('azure.openai.embedding_endpoint')
        embedding_key = config.get('azure.openai.embedding_key')
        embedding_api_version = config.get('azure.openai.embedding_api_version')
        embedding_model = config.get('azure.openai.embedding_model')

        # Display configuration
        print(f"Using endpoint: {embedding_endpoint[:30]}..." if embedding_endpoint else "Embedding endpoint not configured")
        print(f"Using model: {embedding_model}" if embedding_model else "Embedding model not configured")
        print(f"Using API version: {embedding_api_version}" if embedding_api_version else "API version not configured")

        # Check if the embedding endpoint format is complete
        if embedding_endpoint and not embedding_endpoint.endswith("/embeddings"):
            deployment_path = f"openai/deployments/{embedding_model}/embeddings"
            base_endpoint = embedding_endpoint.rstrip("/")
            full_endpoint = f"{base_endpoint}/{deployment_path}"
            print(f"\n⚠️ Adding deployment path to endpoint: {full_endpoint[:50]}...")
        else:
            full_endpoint = embedding_endpoint

        # Extract base endpoint without path
        base_url = embedding_endpoint.split("/openai")[0] if embedding_endpoint and "/openai" in embedding_endpoint else embedding_endpoint

        # Initialize Azure OpenAI client for embeddings
        client = AzureOpenAI(
            azure_endpoint=base_url,
            api_key=embedding_key,
            api_version=embedding_api_version
        )

        # Test text for embedding
        test_text = "This is a test sentence to generate an embedding vector."
        print(f"\n📤 Generating embedding for: '{test_text}'")

        # Get embedding - use model name as the deployment name
        print(f"Calling embeddings API with model: {embedding_model}")
        embedding_response = client.embeddings.create(
            model=embedding_model,
            input=test_text
        )

        # Extract embedding vector
        embedding = embedding_response.data[0].embedding

        # Display embedding summary
        print(f"\n📥 Embedding generated:")
        print(f"  - Dimensions: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        print(f"  - Vector norm: {np.linalg.norm(embedding):.4f}")

        print("\n✅ Azure OpenAI Embeddings test successful!")
        return True

    except Exception as e:
        print(f"\n❌ Azure OpenAI Embeddings test failed: {e}")
        import traceback
        traceback.print_exc()

        # Additional debugging information
        print("\n🔍 Debugging information:")
        print(f"  - Check that your AZURE_OPENAI_ENDPOINT value is correct")
        print(f"  - Verify that AZURE_OPENAI_EMBEDDING_MODEL matches an actual deployment name in your Azure OpenAI service")
        print(f"  - Make sure your embedding deployment is correctly configured in the Azure portal")
        print(f"  - Verify all required environment variables are set in your .env file")

        return False

def main():
    """Run both tests"""
    print("🧪 Testing Azure OpenAI Integration")

    # Run tests
    chat_success = test_azure_openai_chat()
    embedding_success = test_azure_openai_embeddings()

    # Display results
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    print(f"Chat test: {'✅ Passed' if chat_success else '❌ Failed'}")
    print(f"Embedding test: {'✅ Passed' if embedding_success else '❌ Failed'}")

    # Exit with status code
    if chat_success and embedding_success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check configuration and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
