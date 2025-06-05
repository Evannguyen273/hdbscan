"""
Test configuration management and validation
Basic tests without external dependencies
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

def test_config_loads_successfully():
    """Test that configuration loads without errors"""
    try:
        from config.config import Config
        config = Config()
        assert config is not None
        assert hasattr(config, 'bigquery')
        assert hasattr(config, 'azure')
        print("✅ Configuration loads successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration load failed: {e}")
        return False

def test_required_tables_configured():
    """Test that all required BigQuery tables are configured"""
    try:
        from config.config import Config
        config = Config()
        tables = config.bigquery.tables
        
        required_tables = [
            'incident_source', 'preprocessed_incidents', 'predictions', 
            'model_registry', 'training_data', 'cluster_results'
        ]
        
        missing_tables = []
        for table in required_tables:
            if not hasattr(tables, table):
                missing_tables.append(table)
            elif getattr(tables, table) is None:
                missing_tables.append(f"{table} (None value)")
        
        if missing_tables:
            print(f"❌ Missing table configurations: {missing_tables}")
            return False
        else:
            print("✅ All required tables are configured")
            return True
            
    except Exception as e:
        print(f"❌ Table configuration test failed: {e}")
        return False

def test_sql_queries_configured():
    """Test that SQL queries are properly configured"""
    try:
        from config.config import Config
        config = Config()
        queries = config.bigquery.queries
        
        required_queries = [
            'training_data_window', 'model_registry_insert', 'cluster_results_insert'
        ]
        
        missing_queries = []
        for query in required_queries:
            if not hasattr(queries, query):
                missing_queries.append(query)
            else:
                query_text = getattr(queries, query)
                if not query_text or not query_text.strip():
                    missing_queries.append(f"{query} (empty)")
        
        if missing_queries:
            print(f"❌ Missing or empty queries: {missing_queries}")
            return False
        else:
            print("✅ All required SQL queries are configured")
            return True
            
    except Exception as e:
        print(f"❌ SQL queries test failed: {e}")
        return False

def test_environment_variable_substitution():
    """Test environment variable substitution works"""
    try:
        from config.config import Config
        
        # Set a test environment variable
        os.environ['TEST_VAR'] = 'test_value'
        
        config = Config()
        test_config = {
            'test_section': {
                'test_value': '${TEST_VAR}'
            }
        }
        
        substituted = config._substitute_env_vars(test_config)
        
        if substituted['test_section']['test_value'] == 'test_value':
            print("✅ Environment variable substitution works")
            result = True
        else:
            print(f"❌ Environment variable substitution failed: expected 'test_value', got '{substituted['test_section']['test_value']}'")
            result = False
        
        # Cleanup
        del os.environ['TEST_VAR']
        return result
        
    except Exception as e:
        print(f"❌ Environment variable substitution test failed: {e}")
        return False

def test_tech_centers_configuration():
    """Test that tech centers are properly configured"""
    try:
        from config.config import Config
        config = Config()
        tech_centers = config.tech_centers
        
        if not isinstance(tech_centers, list):
            print(f"❌ Tech centers should be a list, got {type(tech_centers)}")
            return False
        
        if len(tech_centers) == 0:
            print("❌ No tech centers configured")
            return False
        
        # Check that tech centers are strings
        invalid_centers = []
        for tc in tech_centers:
            if not isinstance(tc, str) or not tc.strip():
                invalid_centers.append(tc)
        
        if invalid_centers:
            print(f"❌ Invalid tech centers found: {invalid_centers}")
            return False
        else:
            print(f"✅ Tech centers properly configured ({len(tech_centers)} centers)")
            return True
            
    except Exception as e:
        print(f"❌ Tech centers configuration test failed: {e}")
        return False

def test_schema_definitions_exist():
    """Test that schema definitions are properly defined"""
    try:
        from config.schemas import get_bigquery_schema
        
        schema_names = ['incidents', 'preprocessed_incidents', 'cluster_results', 'model_registry']
        
        for schema_name in schema_names:
            schema = get_bigquery_schema(schema_name)
            
            if not isinstance(schema, list):
                print(f"❌ Schema {schema_name} should be a list, got {type(schema)}")
                return False
            
            if len(schema) == 0:
                print(f"❌ Schema {schema_name} is empty")
                return False
            
            # Check that all fields have required properties
            for field in schema:
                if 'name' not in field or 'type' not in field or 'mode' not in field:
                    print(f"❌ Schema {schema_name} field missing required properties: {field}")
                    return False
        
        print("✅ All schema definitions exist and are valid")
        return True
        
    except Exception as e:
        print(f"❌ Schema definitions test failed: {e}")
        return False

def test_incident_schema_validation():
    """Test incident schema validation"""
    try:
        from config.schemas import IncidentSchema
        
        valid_incident = {
            'incident_number': 'INC12345',
            'description': 'Test incident description that is long enough',
            'created_date': '2024-01-15T10:30:00Z',
            'tech_center': 'BT-TC-Data Analytics'
        }
        
        # This should not raise an exception
        incident = IncidentSchema(**valid_incident)
        
        if incident.incident_number == 'INC12345':
            print("✅ Incident schema validation works")
            return True
        else:
            print("❌ Incident schema validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Incident schema validation test failed: {e}")
        return False

def test_operational_tables_configured():
    """Test that operational tables (training_logs, watermarks) are configured"""
    try:
        from config.config import Config
        config = Config()
        tables = config.bigquery.tables
        
        operational_tables = ['training_logs', 'watermarks']
        
        missing_tables = []
        for table in operational_tables:
            if not hasattr(tables, table):
                missing_tables.append(table)
            elif getattr(tables, table) is None:
                missing_tables.append(f"{table} (None value)")
            elif not getattr(tables, table).strip():
                missing_tables.append(f"{table} (empty)")
        
        if missing_tables:
            print(f"❌ Missing operational table configurations: {missing_tables}")
            return False
        else:
            print("✅ All operational tables are configured")
            return True
            
    except Exception as e:
        print(f"❌ Operational tables configuration test failed: {e}")
        return False

def test_new_configuration_sections():
    """Test that new configuration sections are properly loaded"""
    try:
        from config.config import Config
        config = Config()
        
        # Test new sections exist
        new_sections = ['monitoring', 'cost_optimization', 'performance', 'security', 'tech_centers_config']
        missing_sections = []
        
        for section in new_sections:
            if not hasattr(config, section):
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing configuration sections: {missing_sections}")
            return False
        else:
            print("✅ All new configuration sections are accessible")
            return True
            
    except Exception as e:
        print(f"❌ New configuration sections test failed: {e}")
        return False

def test_text_processing_large_input_handling():
    """Test that text processing can handle very large inputs"""
    try:
        from config.config import Config
        from preprocessing.text_processing import TextProcessor
        import pandas as pd
        
        config = Config()
        processor = TextProcessor(config)
        
        # Create a very large text (simulate 500K characters)
        large_text = "This is a test incident description. " * 15000  # ~500K chars
        
        # Test chunking logic
        prompt = processor._build_summarization_prompt(large_text)
        
        # Should trigger chunking since 500K > 400K limit
        if "SECTION BREAK" in prompt:
            print("✅ Large input chunking works correctly")
            return True
        else:
            print("❌ Large input chunking failed")
            return False
            
    except Exception as e:
        print(f"❌ Large input handling test failed: {e}")
        return False

def test_text_processing_medium_input_truncation():
    """Test that medium-sized inputs are truncated correctly"""
    try:
        from config.config import Config  
        from preprocessing.text_processing import TextProcessor
        
        config = Config()
        processor = TextProcessor(config)
        
        # Create medium text (60K characters)
        medium_text = "This is a medium incident description. " * 1500  # ~60K chars
        
        # Test truncation logic
        prompt = processor._build_summarization_prompt(medium_text)
        
        # Should trigger truncation since 60K > 50K but < 400K
        if "content truncated" in prompt:
            print("✅ Medium input truncation works correctly")
            return True
        else:
            print("❌ Medium input truncation failed")
            return False
            
    except Exception as e:
        print(f"❌ Medium input handling test failed: {e}")
        return False

def test_text_processing_token_limits_config():
    """Test that token limit configurations are properly loaded"""
    try:
        from config.config import Config
        
        config = Config()
        
        # Check new summarization config
        summarization = config.preprocessing.summarization
        
        if not hasattr(summarization, 'max_input_chars'):
            print("❌ max_input_chars not configured")
            return False
            
        if not hasattr(summarization, 'chunk_size_chars'):
            print("❌ chunk_size_chars not configured")
            return False
            
        # Validate values
        if summarization.max_input_chars != 400000:
            print(f"❌ max_input_chars should be 400000, got {summarization.max_input_chars}")
            return False
            
        if summarization.chunk_size_chars != 50000:
            print(f"❌ chunk_size_chars should be 50000, got {summarization.chunk_size_chars}")
            return False
            
        print("✅ Token limit configurations loaded correctly")
        return True
        
    except Exception as e:
        print(f"❌ Token limits config test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and return summary"""
    tests = [
        test_config_loads_successfully,
        test_required_tables_configured,
        test_sql_queries_configured,
        test_environment_variable_substitution,
        test_tech_centers_configuration,
        test_schema_definitions_exist,
        test_incident_schema_validation,
        test_operational_tables_configured,
        test_new_configuration_sections,
        test_text_processing_large_input_handling,
        test_text_processing_medium_input_truncation,
        test_text_processing_token_limits_config
    ]
    
    print("Running configuration and schema validation tests...\n")
    
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
        print()  # Empty line for readability
    
    print("=" * 50)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)