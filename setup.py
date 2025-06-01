#!/usr/bin/env python3
"""
Setup script for HDBSCAN Clustering Pipeline
Copies template files and guides user through configuration
"""

import os
import shutil
from pathlib import Path

def setup_configuration():
    """Setup configuration files from templates"""
    print("🔧 Setting up HDBSCAN Clustering Pipeline configuration...")
    
    # Define file mappings
    file_mappings = [
        ('config/config_template.py', 'config/config.py'),
        ('config/config_template.yaml', 'config/config.yaml'),
        ('.env.example', '.env')
    ]
    
    created_files = []
    
    for template, target in file_mappings:
        template_path = Path(template)
        target_path = Path(target)
        
        if not template_path.exists():
            print(f"❌ Template file not found: {template}")
            continue
            
        if target_path.exists():
            response = input(f"⚠️  {target} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print(f"   Skipping {target}")
                continue
        
        # Copy template to target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template_path, target_path)
        created_files.append(str(target_path))
        print(f"✅ Created: {target}")
    
    if created_files:
        print(f"\n🎉 Successfully created {len(created_files)} configuration files!")
        print("\n📝 Next steps:")
        print("   1. Edit .env with your actual credentials")
        print("   2. Edit config/config.yaml and replace YOUR_PROJECT_ID placeholders")
        print("   3. Run: python test_config.py")
        print("   4. Run: python main.py validate")
        
        print(f"\n🔒 Security note:")
        print("   These files are ignored by git and will not be committed:")
        for file in created_files:
            print(f"   - {file}")
            
    else:
        print("❌ No files were created")

def check_git_status():
    """Check if sensitive files are properly ignored"""
    print("\n🔍 Checking git ignore status...")
    
    sensitive_files = [
        '.env',
        'config/config.py', 
        'config/config.yaml'
    ]
    
    for file in sensitive_files:
        if Path(file).exists():
            # Check if file is ignored by git
            result = os.system(f'git check-ignore {file} > /dev/null 2>&1')
            if result == 0:
                print(f"✅ {file} is properly ignored by git")
            else:
                print(f"⚠️  {file} is NOT ignored by git - this is a security risk!")
        else:
            print(f"ℹ️  {file} does not exist yet")

def main():
    """Main setup function"""
    print("🚀 HDBSCAN Clustering Pipeline Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('config').exists():
        print("❌ Error: config directory not found!")
        print("   Please run this script from the project root directory")
        return
    
    # Setup configuration
    setup_configuration()
    
    # Check git status
    check_git_status()
    
    print("\n" + "=" * 50)
    print("🎯 Setup complete! You can now configure your pipeline.")

if __name__ == "__main__":
    main()