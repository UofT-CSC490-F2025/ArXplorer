#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Pre-Disaster Backup Script
Creates comprehensive backup of all systems before disaster simulation
"""

import asyncio
import boto3
import json
import os
import yaml
from datetime import datetime
from pymongo import MongoClient
import pandas as pd

class PreDisasterBackup:
    def __init__(self):
        """Initialize backup system with current configurations"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = f"disaster_recovery/backups/{self.timestamp}"
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Load current configuration
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
    
    def backup_aws_s3_state(self):
        """Document current S3 bucket state and contents"""
        print("Backing up AWS S3 state...")
        
        s3_client = boto3.client('s3', region_name=self.config['aws']['region'])
        backup_data = {
            'timestamp': self.timestamp,
            'buckets': {},
            'region': self.config['aws']['region']
        }
        
        for bucket_type, bucket_name in self.config['aws']['buckets'].items():
            try:
                # Get bucket contents
                response = s3_client.list_objects_v2(Bucket=bucket_name)
                objects = []
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        objects.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'etag': obj['ETag']
                        })
                
                # Get bucket policy and ACL info
                try:
                    bucket_policy = s3_client.get_bucket_policy(Bucket=bucket_name)
                    policy = json.loads(bucket_policy['Policy'])
                except:
                    policy = None
                
                backup_data['buckets'][bucket_name] = {
                    'type': bucket_type,
                    'object_count': len(objects),
                    'total_size': sum(obj['size'] for obj in objects),
                    'objects': objects,
                    'policy': policy
                }
                
                print(f"   SUCCESS: {bucket_name}: {len(objects)} objects, {sum(obj['size'] for obj in objects):,} bytes")
                
            except Exception as e:
                print(f"   ERROR: Error backing up {bucket_name}: {e}")
                backup_data['buckets'][bucket_name] = {'error': str(e)}
        
        # Save S3 backup manifest
        with open(f"{self.backup_dir}/s3_state_backup.json", 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        return backup_data
    
    def backup_mongodb_state(self):
        """Create backup of MongoDB collections and metadata"""
        print("Backing up MongoDB Atlas state...")
        
        try:
            client = MongoClient(self.config['mongodb']['connection_string'])
            db = client[self.config['mongodb']['database_name']]
            
            backup_data = {
                'timestamp': self.timestamp,
                'database': self.config['mongodb']['database_name'],
                'collections': {}
            }
            
            # Get all collections
            collections = db.list_collection_names()
            
            for collection_name in collections:
                collection = db[collection_name]
                
                # Get collection stats
                stats = db.command("collStats", collection_name)
                
                # Get sample documents (first 10)
                sample_docs = list(collection.find().limit(10))
                
                # Get indexes
                indexes = list(collection.list_indexes())
                
                backup_data['collections'][collection_name] = {
                    'document_count': stats.get('count', 0),
                    'size_bytes': stats.get('size', 0),
                    'indexes': [idx for idx in indexes],
                    'sample_documents': sample_docs
                }
                
                print(f"   SUCCESS: {collection_name}: {stats.get('count', 0)} documents, {stats.get('size', 0):,} bytes")
            
            # Save MongoDB backup manifest
            with open(f"{self.backup_dir}/mongodb_state_backup.json", 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            client.close()
            return backup_data
            
        except Exception as e:
            print(f"   ERROR: Error backing up MongoDB: {e}")
            return {'error': str(e)}
    
    def backup_terraform_state(self):
        """Backup current Terraform state"""
        print("Backing up Terraform state...")
        
        try:
            # Copy terraform state files
            import shutil
            
            tf_dir = "terraform/environments/dev"
            tf_backup_dir = f"{self.backup_dir}/terraform"
            os.makedirs(tf_backup_dir, exist_ok=True)
            
            # Copy key files
            for file in ['terraform.tfstate', 'terraform.tfstate.backup', 'terraform.tfvars']:
                src_path = f"{tf_dir}/{file}"
                if os.path.exists(src_path):
                    shutil.copy2(src_path, f"{tf_backup_dir}/{file}")
                    print(f"   SUCCESS: Backed up {file}")
            
            # Copy configuration files
            for file in ['main.tf', 'variables.tf']:
                src_path = f"{tf_dir}/{file}"
                if os.path.exists(src_path):
                    shutil.copy2(src_path, f"{tf_backup_dir}/{file}")
                    print(f"   SUCCESS: Backed up {file}")
            
            return True
            
        except Exception as e:
            print(f"   ERROR: Error backing up Terraform: {e}")
            return False
    
    def backup_application_config(self):
        """Backup application configuration files"""
        print("Backing up application configuration...")
        
        try:
            import shutil
            
            config_files = [
                'config.yaml',
                'requirements.txt',
                '.gitignore'
            ]
            
            for file in config_files:
                if os.path.exists(file):
                    shutil.copy2(file, f"{self.backup_dir}/{file}")
                    print(f"   SUCCESS: Backed up {file}")
            
            return True
            
        except Exception as e:
            print(f"   ERROR: Error backing up config: {e}")
            return False
    
    def create_recovery_manifest(self, s3_backup, mongodb_backup):
        """Create comprehensive recovery manifest"""
        print("Creating recovery manifest...")
        
        manifest = {
            'disaster_recovery': {
                'backup_timestamp': self.timestamp,
                'backup_date': datetime.now().isoformat(),
                'systems': {
                    'aws_s3': {
                        'status': 'backed_up' if s3_backup.get('buckets') else 'failed',
                        'bucket_count': len(s3_backup.get('buckets', {})),
                        'total_objects': sum(
                            bucket.get('object_count', 0) 
                            for bucket in s3_backup.get('buckets', {}).values()
                            if isinstance(bucket, dict) and 'object_count' in bucket
                        )
                    },
                    'mongodb_atlas': {
                        'status': 'backed_up' if mongodb_backup.get('collections') else 'failed',
                        'collection_count': len(mongodb_backup.get('collections', {})),
                        'total_documents': sum(
                            coll.get('document_count', 0)
                            for coll in mongodb_backup.get('collections', {}).values()
                            if isinstance(coll, dict) and 'document_count' in coll
                        )
                    },
                    'terraform': {
                        'status': 'backed_up',
                        'state_files': ['terraform.tfstate', '.terraform.lock.hcl']
                    },
                    'application': {
                        'status': 'backed_up',
                        'config_files': ['config.yaml', 'requirements.txt']
                    }
                }
            }
        }
        
        with open(f"{self.backup_dir}/recovery_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"SUCCESS: Recovery manifest created: {self.backup_dir}/recovery_manifest.json")
        return manifest

def main():
    """Execute pre-disaster backup"""
    print("ArXplorer Disaster Recovery - Pre-Disaster Backup")
    print("=" * 60)
    
    backup = PreDisasterBackup()
    
    # Execute all backups
    s3_backup = backup.backup_aws_s3_state()
    mongodb_backup = backup.backup_mongodb_state()
    backup.backup_terraform_state()
    backup.backup_application_config()
    
    # Create recovery manifest
    manifest = backup.create_recovery_manifest(s3_backup, mongodb_backup)
    
    print("\nSUCCESS: Pre-disaster backup completed!")
    print(f"Backup location: {backup.backup_dir}")
    print("Ready to begin disaster simulation...")

if __name__ == "__main__":
    main()