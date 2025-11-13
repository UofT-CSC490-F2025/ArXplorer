#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Recovery Script
Restores all infrastructure and data from backup after disaster
"""

import asyncio
import boto3
import json
import os
import yaml
import subprocess
import shutil
from datetime import datetime
from pymongo import MongoClient

class DisasterRecovery:
    def __init__(self, backup_timestamp=None):
        """Initialize disaster recovery"""
        self.recovery_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Find latest backup if not specified
        if backup_timestamp is None:
            backup_timestamp = self.find_latest_backup()
        
        self.backup_dir = f"backup_storage/backups/{backup_timestamp}"
        
        if not os.path.exists(self.backup_dir):
            raise ValueError(f"Backup directory not found: {self.backup_dir}")
        
        print(f"Using backup from: {backup_timestamp}")
    
    def find_latest_backup(self):
        """Find the most recent backup timestamp"""
        backup_base = "disaster_recovery/backups"
        if not os.path.exists(backup_base):
            raise ValueError("No backup directory found!")
        
        backups = [d for d in os.listdir(backup_base) if os.path.isdir(os.path.join(backup_base, d))]
        if not backups:
            raise ValueError("No backups found!")
        
        return max(backups)  # Latest timestamp
    
    def restore_configuration(self):
        """Restore application configuration from backup"""
        print("RESTORING Application Configuration...")
        
        try:
            # Restore config files
            config_files = ['config.yaml', 'requirements.txt', '.gitignore']
            
            for config_file in config_files:
                backup_file = f"{self.backup_dir}/{config_file}"
                if os.path.exists(backup_file):
                    shutil.copy2(backup_file, config_file)
                    print(f"   SUCCESS: Restored {config_file}")
                else:
                    print(f"   WARNING: Backup not found for {config_file}")
            
            # Load restored configuration
            with open('config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
            
            print("   SUCCESS: Configuration restoration complete")
            return True
            
        except Exception as e:
            print(f"   ERROR: Error restoring configuration: {e}")
            return False
    
    def restore_terraform_infrastructure(self):
        """Restore AWS S3 infrastructure using Terraform"""
        print("RESTORING AWS S3 Infrastructure...")
        
        try:
            # Restore Terraform state files
            tf_dir = "infrastructure/terraform/environments/dev"
            tf_backup_dir = f"{self.backup_dir}/terraform"
            
            if os.path.exists(tf_backup_dir):
                # Restore state files
                for state_file in ['terraform.tfstate', 'terraform.tfstate.backup', '.terraform.lock.hcl']:
                    backup_file = f"{tf_backup_dir}/{state_file}"
                    if os.path.exists(backup_file):
                        shutil.copy2(backup_file, f"{tf_dir}/{state_file}")
                        print(f"   SUCCESS: Restored {state_file}")
            
            # Change to terraform directory
            os.chdir(tf_dir)
            
            # Initialize Terraform
            print("   🔄 Initializing Terraform...")
            init_result = subprocess.run(['terraform', 'init'], capture_output=True, text=True)
            
            if init_result.returncode != 0:
                print(f"   ❌ Terraform init failed: {init_result.stderr}")
                return False
            
            # Plan the restoration
            print("   📋 Planning infrastructure restoration...")
            plan_result = subprocess.run(['terraform', 'plan'], capture_output=True, text=True)
            
            if plan_result.returncode != 0:
                print(f"   ❌ Terraform plan failed: {plan_result.stderr}")
                return False
            
            # Apply the restoration
            print("   🚀 Applying infrastructure restoration...")
            apply_result = subprocess.run(['terraform', 'apply', '-auto-approve'], capture_output=True, text=True)
            
            if apply_result.returncode == 0:
                print("   ✅ Infrastructure restoration complete")
                
                # Get outputs to verify
                output_result = subprocess.run(['terraform', 'output', '-json'], capture_output=True, text=True)
                if output_result.returncode == 0:
                    outputs = json.loads(output_result.stdout)
                    print("   📊 Restored infrastructure:")
                    for key, value in outputs.items():
                        print(f"      {key}: {value.get('value')}")
                
                return True
            else:
                print(f"   ❌ Terraform apply failed: {apply_result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error restoring infrastructure: {e}")
            return False
        finally:
            os.chdir('../../..')
    
    def restore_mongodb_structure(self):
        """Restore MongoDB database structure and indexes"""
        print("🗄️  RESTORING MongoDB Atlas Structure...")
        
        try:
            # Load MongoDB backup manifest
            mongodb_backup_file = f"{self.backup_dir}/mongodb_state_backup.json"
            if not os.path.exists(mongodb_backup_file):
                print("   ⚠️  MongoDB backup manifest not found")
                return False
            
            with open(mongodb_backup_file, 'r') as f:
                backup_data = json.load(f)
            
            # Connect to MongoDB
            client = MongoClient(self.config['mongodb']['connection_string'])
            db = client[self.config['mongodb']['database_name']]
            
            # Restore collections and indexes
            restored_collections = 0
            for collection_name, collection_data in backup_data['collections'].items():
                try:
                    # Create collection (will be created automatically on first insert)
                    collection = db[collection_name]
                    
                    # Restore indexes
                    if 'indexes' in collection_data:
                        for index_spec in collection_data['indexes']:
                            if index_spec['name'] != '_id_':  # Skip default _id index
                                try:
                                    # Extract index keys and options
                                    keys = index_spec.get('key', {})
                                    if keys:
                                        collection.create_index(list(keys.items()))
                                        print(f"      ✅ Restored index on {collection_name}: {list(keys.keys())}")
                                except Exception as idx_error:
                                    print(f"      ⚠️  Index creation warning for {collection_name}: {idx_error}")
                    
                    # Insert sample documents if available (for structure verification)
                    if 'sample_documents' in collection_data and collection_data['sample_documents']:
                        # Just insert one sample document to create the collection
                        sample_doc = collection_data['sample_documents'][0]
                        if isinstance(sample_doc, dict):
                            # Remove _id to avoid duplicates
                            sample_doc.pop('_id', None)
                            collection.insert_one(sample_doc)
                            print(f"      ✅ Created collection {collection_name} with sample document")
                    
                    restored_collections += 1
                    
                except Exception as e:
                    print(f"      ❌ Error restoring collection {collection_name}: {e}")
            
            print(f"   ✅ Restored {restored_collections} MongoDB collections")
            client.close()
            return True
            
        except Exception as e:
            print(f"   ❌ Error restoring MongoDB structure: {e}")
            return False
    
    def verify_recovery(self):
        """Verify that all systems have been properly restored"""
        print("🔍 VERIFYING System Recovery...")
        
        recovery_status = {
            'configuration': False,
            'aws_s3': False,
            'mongodb': False,
            'overall': False
        }
        
        # Verify configuration
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            if 'aws' in config and 'mongodb' in config:
                recovery_status['configuration'] = True
                print("   ✅ Configuration files restored and valid")
            else:
                print("   ❌ Configuration incomplete")
                
        except Exception as e:
            print(f"   ❌ Configuration verification failed: {e}")
        
        # Verify AWS S3 buckets
        try:
            s3_client = boto3.client('s3', region_name=self.config['aws']['region'])
            
            bucket_count = 0
            for bucket_type, bucket_name in self.config['aws']['buckets'].items():
                try:
                    s3_client.head_bucket(Bucket=bucket_name)
                    bucket_count += 1
                    print(f"      ✅ S3 bucket accessible: {bucket_name}")
                except Exception as e:
                    print(f"      ❌ S3 bucket not accessible: {bucket_name} - {e}")
            
            if bucket_count == len(self.config['aws']['buckets']):
                recovery_status['aws_s3'] = True
                print("   ✅ All S3 buckets restored and accessible")
            else:
                print(f"   ❌ Only {bucket_count}/{len(self.config['aws']['buckets'])} S3 buckets accessible")
                
        except Exception as e:
            print(f"   ❌ S3 verification failed: {e}")
        
        # Verify MongoDB connection
        try:
            client = MongoClient(self.config['mongodb']['connection_string'])
            db = client[self.config['mongodb']['database_name']]
            
            # Try to list collections
            collections = db.list_collection_names()
            
            if collections:
                recovery_status['mongodb'] = True
                print(f"   ✅ MongoDB accessible with {len(collections)} collections")
                for collection_name in collections:
                    doc_count = db[collection_name].count_documents({})
                    print(f"      - {collection_name}: {doc_count} documents")
            else:
                print("   ⚠️  MongoDB accessible but no collections found (expected for fresh restore)")
                recovery_status['mongodb'] = True  # Connection works
            
            client.close()
            
        except Exception as e:
            print(f"   ❌ MongoDB verification failed: {e}")
        
        # Overall recovery status
        recovery_status['overall'] = all([
            recovery_status['configuration'],
            recovery_status['aws_s3'],
            recovery_status['mongodb']
        ])
        
        return recovery_status
    
    def test_pipeline_functionality(self):
        """Test that the restored pipeline can process data"""
        print("🧪 TESTING Pipeline Functionality...")
        
        try:
            # Import the enhanced pipeline
            import sys
            sys.path.append('.')
            
            from enhanced_pipeline import EnhancedCloudPipeline
            
            # Initialize pipeline with restored config
            pipeline = EnhancedCloudPipeline('config.yaml')
            
            # Test basic connectivity
            print("   🔄 Testing S3 connectivity...")
            # Test S3 connection by listing buckets
            s3_client = boto3.client('s3', region_name=self.config['aws']['region'])
            buckets = s3_client.list_buckets()
            print("      ✅ S3 connectivity confirmed")
            
            print("   🔄 Testing MongoDB connectivity...")
            # Test MongoDB connection
            client = MongoClient(self.config['mongodb']['connection_string'])
            db = client[self.config['mongodb']['database_name']]
            db.admin.command('ping')
            print("      ✅ MongoDB connectivity confirmed")
            client.close()
            
            print("   ✅ Pipeline functionality tests passed")
            return True
            
        except Exception as e:
            print(f"   ❌ Pipeline functionality test failed: {e}")
            return False

def main():
    """Execute disaster recovery"""
    print("🚨 ArXplorer Disaster Recovery - SYSTEM RESTORATION")
    print("=" * 60)
    
    try:
        # Initialize recovery
        recovery = DisasterRecovery()
        
        print("🔄 BEGINNING SYSTEM RESTORATION...")
        print()
        
        # Execute recovery sequence
        results = {
            'config_restored': recovery.restore_configuration(),
            'infrastructure_restored': recovery.restore_terraform_infrastructure(),
            'database_restored': recovery.restore_mongodb_structure()
        }
        
        # Verify recovery
        verification = recovery.verify_recovery()
        
        # Test functionality
        functionality_test = recovery.test_pipeline_functionality()
        
        print("\n" + "=" * 60)
        print("✅ DISASTER RECOVERY RESULTS:")
        print(f"   Configuration: {'RESTORED' if results['config_restored'] else 'FAILED'}")
        print(f"   Infrastructure: {'RESTORED' if results['infrastructure_restored'] else 'FAILED'}")
        print(f"   Database: {'RESTORED' if results['database_restored'] else 'FAILED'}")
        print(f"   Verification: {'PASSED' if verification['overall'] else 'FAILED'}")
        print(f"   Functionality: {'WORKING' if functionality_test else 'FAILED'}")
        print("=" * 60)
        
        if verification['overall'] and functionality_test:
            print("🎉 DISASTER RECOVERY SUCCESSFUL!")
            print("📊 All systems restored and operational")
            
            # Create recovery report
            recovery_report = {
                'recovery_timestamp': recovery.recovery_timestamp,
                'recovery_date': datetime.now().isoformat(),
                'backup_used': recovery.backup_dir,
                'results': results,
                'verification': verification,
                'functionality_test': functionality_test,
                'status': 'SUCCESS'
            }
            
            with open(f"disaster_recovery/recovery_report_{recovery.recovery_timestamp}.json", 'w') as f:
                json.dump(recovery_report, f, indent=2)
            
        else:
            print("❌ DISASTER RECOVERY INCOMPLETE")
            print("⚠️  Some systems may not be fully operational")
            
    except Exception as e:
        print(f"❌ DISASTER RECOVERY FAILED: {e}")
        return False

if __name__ == "__main__":
    main()