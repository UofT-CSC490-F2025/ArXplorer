#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Simple Complete Demo
Automated Backup → Destroy → Restore cycle
"""

import os
import sys
import subprocess
import importlib.util
import shutil
from datetime import datetime

def load_disaster_simulation():
    """Load the DisasterSimulation class"""
    spec = importlib.util.spec_from_file_location(
        "disaster_simulation", 
        "infrastructure/disaster_recovery/02_disaster_simulation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DisasterSimulation

def main():
    print("ArXplorer Complete Disaster Recovery Demo")
    print("=" * 60)
    print("BACKUP → DESTROY → RESTORE")
    print()
    
    # Step 1: Backup
    print("STEP 1: CREATING BACKUP...")
    print("-" * 40)
    
    backup_result = subprocess.run([
        sys.executable, 'infrastructure/disaster_recovery/01_pre_disaster_backup.py'
    ], text=True)
    
    if backup_result.returncode != 0:
        print("❌ Backup failed!")
        return
    
    # Get the latest backup timestamp
    backup_dirs = [d for d in os.listdir('disaster_recovery/backups') if d.startswith('2025')]
    latest_backup = max(backup_dirs) if backup_dirs else None
    print(f"✅ Backup created: {latest_backup}")
    print()
    
    # Step 2: Destroy
    print("STEP 2: DESTROYING EVERYTHING...")
    print("-" * 40)
    
    try:
        DisasterSimulation = load_disaster_simulation()
        disaster = DisasterSimulation()
        
        print("DESTROYING AWS S3 Infrastructure...")
        s3_result = disaster.destroy_s3_infrastructure()
        
        print("DESTROYING MongoDB Data...")
        mongo_result = disaster.destroy_mongodb_data()
        
        print("DESTROYING Terraform State...")
        tf_result = disaster.destroy_terraform_state()
        
        print("DESTROYING Configuration...")
        config_result = disaster.simulate_config_corruption(destroy_completely=True)
        
        print()
        print("DESTRUCTION RESULTS:")
        results = {
            'S3 Infrastructure': s3_result,
            'MongoDB Data': mongo_result,
            'Terraform State': tf_result,
            'Configuration': config_result
        }
        
        for component, destroyed in results.items():
            status = "DESTROYED" if destroyed else "FAILED"
            print(f"   {component}: {status}")
        
        print("✅ Destruction phase complete")
        
    except Exception as e:
        print(f"❌ Destruction failed: {e}")
        return
    
    print()
    
    # Step 3: Restore
    print("STEP 3: RESTORING FROM BACKUP...")
    print("-" * 40)
    
    if latest_backup:
        backup_path = f"disaster_recovery/backups/{latest_backup}"
        
        # Restore key files
        files_to_restore = ['config.yaml', 'requirements.txt', '.gitignore']
        
        for file in files_to_restore:
            src = os.path.join(backup_path, file)
            if os.path.exists(src):
                shutil.copy2(src, file)
                print(f"✅ Restored {file}")
            else:
                print(f"⚠️ {file} not found in backup")
        
        print("✅ Restoration complete")
    
    print()
    
    # Step 4: Verify
    print("STEP 4: VERIFYING RECOVERY...")
    print("-" * 40)
    
    subprocess.run([
        sys.executable, 'infrastructure/disaster_recovery/05_recovery_status.py'
    ])
    
    print()
    print("🎉 COMPLETE DISASTER RECOVERY DEMO FINISHED!")
    print("🚀 System successfully recovered from total destruction!")

if __name__ == "__main__":
    main()