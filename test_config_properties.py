#!/usr/bin/env python3
"""
Config property testing script - shows how to import and test config properties
"""

def test_config_properties():
    """Test and demonstrate config property access"""
    print("🔧 Testing Config Property Access...")
    print("=" * 50)
    
    try:
        # First, copy templates if they don't exist
        from pathlib import Path
        
        if not Path('config/config.py').exists():
            print("❌ config/config.py not found!")
            print("   Run: cp config/config_template.py config/config.py")
            return False
            
        if not Path('config/config.yaml').exists():
            print("❌ config/config.yaml not found!")
            print("   Run: cp config/config_template.yaml config/config.yaml")
            return False
        
        # Import and load config
        from config.config import get_config, load_config
        
        print("📥 Loading configuration...")
        config = get_config()
        
        # Test BigQuery properties
        print("\n🗄️ BigQuery Configuration:")
        print(f"   Project ID: {config.bigquery.project_id}")
        print(f"   Service Account Type: {type(config.bigquery.service_account_key_path)}")
        
        # Test table access
        tables = config.bigquery.tables
        print(f"   Incident Table: {tables.incidents}")
        print(f"   Team Services Table: {tables.team_services}")
        print(f"   Problems Table: {tables.problems}")
        
        # Test Azure properties
        print("\n☁️ Azure Configuration:")
        azure = config.azure.openai
        print(f"   Chat Endpoint: {azure.endpoint[:30]}...")
        print(f"   Deployment: {azure.deployment_name}")
        print(f"   Embedding Model: {azure.embedding_model}")
        
        # Test clustering properties
        print("\n🧠 Clustering Configuration:")
        clustering = config.clustering
        print(f"   HDBSCAN min_cluster_size: {clustering.hdbscan.min_cluster_size}")
        print(f"   UMAP n_components: {clustering.umap.n_components}")
        
        # Test embedding weights
        weights = clustering.embedding.weights
        print(f"   Semantic weight: {weights.semantic}")
        print(f"   Entity weight: {weights.entity}")
        print(f"   Action weight: {weights.action}")
        
        # Test tech centers
        print("\n🏢 Tech Centers:")
        tech_centers = config.tech_centers
        print(f"   Total centers: {len(tech_centers)}")
        print(f"   First center: {tech_centers[0] if tech_centers else 'None'}")
        
        # Test blob storage
        print("\n☁️ Blob Storage:")
        blob = config.get('blob_storage', {})
        print(f"   Container: {blob.get('container_name', 'N/A')}")
        print(f"   Connection: {blob.get('connection_string', 'N/A')[:30]}...")
        
        # Test dot notation access
        print("\n🔍 Dot Notation Access:")
        print(f"   clustering.embedding.batch_size: {config.get('clustering.embedding.batch_size')}")
        print(f"   pipeline.max_workers: {config.get('pipeline.max_workers')}")
        
        print("\n✅ All config properties accessible!")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_config_usage():
    """Show practical config usage examples"""
    print("\n" + "=" * 50)
    print("📚 Practical Config Usage Examples:")
    
    from config.config import get_config
    config = get_config()
    
    # Example 1: BigQuery client setup
    print("\n1️⃣ BigQuery Client Setup:")
    print("```python")
    print("from google.cloud import bigquery")
    print("from config.config import get_config")
    print("")
    print("config = get_config()")
    print("client = bigquery.Client.from_service_account_info(")
    print("    config.bigquery.service_account_key_path,")
    print(f"    project='{config.bigquery.project_id}'")
    print(")")
    print("```")
    
    # Example 2: Azure OpenAI setup
    print("\n2️⃣ Azure OpenAI Setup:")
    print("```python")
    print("from openai import AzureOpenAI")
    print("from config.config import get_config")
    print("")
    print("config = get_config()")
    print("client = AzureOpenAI(")
    print(f"    azure_endpoint=config.azure.openai.endpoint,")
    print(f"    api_key=config.azure.openai.api_key,")
    print(f"    api_version=config.azure.openai.api_version")
    print(")")
    print("```")
    
    # Example 3: HDBSCAN parameters
    print("\n3️⃣ HDBSCAN Configuration:")
    print("```python")
    print("from hdbscan import HDBSCAN")
    print("from config.config import get_config")
    print("")
    print("config = get_config()")
    print("clusterer = HDBSCAN(")
    print(f"    min_cluster_size={config.clustering.hdbscan.min_cluster_size},")
    print(f"    min_samples={config.clustering.hdbscan.min_samples}")
    print(")")
    print("```")

if __name__ == "__main__":
    print("🧪 Config Property Testing and Examples")
    
    # Test config properties
    if test_config_properties():
        # Show usage examples
        demonstrate_config_usage()
        
        print("\n" + "=" * 50)
        print("🎉 Config system working perfectly!")
        print("\n🚀 Next steps:")
        print("   1. Run: python test_simple.py")
        print("   2. Run: python main.py validate")
    else:
        print("\n❌ Please fix config issues first")
        print("   1. Copy templates: cp config/config_template.* config/")
        print("   2. Edit config/config.yaml with your values")
        print("   3. Create .env file with credentials")