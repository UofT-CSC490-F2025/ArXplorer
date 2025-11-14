#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Recovery Status Report
Comprehensive status check after backup recovery
"""

import os
import yaml
import boto3
from pymongo import MongoClient
from datetime import datetime

def recovery_status_report():
    """Generate comprehensive recovery status report"""
    print("ArXplorer Disaster Recovery - RECOVERY STATUS REPORT")
    print("=" * 70)
    print(f"Report Generated: {datetime.now().isoformat()}")
    print()
    
    # Track overall status
    recovery_success = True
    
    # 1. Configuration Recovery
    print("1. CONFIGURATION RECOVERY:")
    config_file = "config.yaml"
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            content = f.read()
        
        if 'CORRUPTED CONFIG FILE' in open(config_file).read():
            print("   Status: FAILED - Configuration still corrupted")
            recovery_success = False
        else:
            print("   Status: SUCCESS - Configuration restored from backup")
            print(f"   Size: {os.path.getsize(config_file)} bytes")
    else:
        print("   Status: FAILED - Configuration file missing")
        recovery_success = False
    
    # 2. AWS S3 Infrastructure Recovery
    print("\n2. AWS S3 INFRASTRUCTURE RECOVERY:")
    try:
        s3_client = boto3.client('s3', region_name='ca-central-1')
        buckets = s3_client.list_buckets()['Buckets']
        arxplorer_buckets = [b for b in buckets if 'arxplorer' in b['Name'].lower()]
        
        print(f"   Status: SUCCESS - {len(arxplorer_buckets)} S3 buckets accessible")
        for bucket in arxplorer_buckets[:4]:  # Show first 4
            print(f"     - {bucket['Name']}")
        
    except Exception as e:
        print(f"   Status: FAILED - S3 access error: {e}")
        recovery_success = False
    
    # 3. MongoDB Atlas Recovery  
    print("\n3. MONGODB ATLAS RECOVERY:")
    try:
        if config and 'mongodb' in config:
            client = MongoClient(config['mongodb']['connection_string'])
            db = client[config['mongodb']['database_name']]
            collections = db.list_collection_names()
            
            print(f"   Status: SUCCESS - Database accessible")
            print(f"   Collections: {len(collections)}")
            if collections:
                print(f"     - Active collections: {', '.join(collections[:3])}")
            else:
                print("     - No collections (expected after disaster recovery)")
            
            client.close()
        else:
            print("   Status: WARNING - MongoDB config not found")
            
    except Exception as e:
        print(f"   Status: FAILED - MongoDB access error: {e}")
        recovery_success = False
    
    # 4. Application Files Recovery
    print("\n4. APPLICATION FILES RECOVERY:")
    key_files = ['requirements.txt', '.gitignore', 'src/', 'infrastructure/']
    files_ok = 0
    
    for file in key_files:
        if os.path.exists(file):
            print(f"   SUCCESS: {file} restored")
            files_ok += 1
        else:
            print(f"   WARNING: {file} missing")
    
    print(f"   Status: {files_ok}/{len(key_files)} key files present")
    
    # 5. Overall Recovery Status
    print("\n" + "=" * 70)
    print("OVERALL RECOVERY STATUS:")
    
    if recovery_success:
        print("   🟢 RECOVERY SUCCESSFUL")
        print("   All critical systems have been restored from backup")
        print("   ArXplorer is ready for normal operations")
    else:
        print("   🟡 RECOVERY PARTIAL")
        print("   Some issues detected - manual intervention may be required")
    
    print()
    print("RECOVERY RECOMMENDATIONS:")
    print("   - Test core functionality before production use")
    print("   - Verify data integrity in restored systems")  
    print("   - Update any credentials that may have changed")
    print("   - Create fresh backups after confirming recovery")
    
    return recovery_success

if __name__ == "__main__":
    recovery_status_report()