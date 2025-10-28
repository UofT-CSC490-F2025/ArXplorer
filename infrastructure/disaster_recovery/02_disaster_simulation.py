#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Disaster Simulation Script
Systematically destroys all infrastructure and data to simulate disaster
"""

import asyncio
import boto3
import json
import os
import yaml
import subprocess
from datetime import datetime
from pymongo import MongoClient

class DisasterSimulation:
    def __init__(self):
        """Initialize disaster simulation"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load current configuration
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
    
    def destroy_s3_infrastructure(self):
        """Destroy all S3 buckets and data using Terraform"""
        print("💥 DESTROYING AWS S3 Infrastructure...")
        
        try:
            # First, empty all buckets (Terraform can't destroy non-empty buckets)
            s3_client = boto3.client('s3', region_name=self.config['aws']['region'])
            
            for bucket_type, bucket_name in self.config['aws']['buckets'].items():
                try:
                    print(f"   🗑️  Emptying bucket: {bucket_name}")
                    
                    # List and delete all objects
                    response = s3_client.list_objects_v2(Bucket=bucket_name)
                    
                    if 'Contents' in response:
                        objects = [{'Key': obj['Key']} for obj in response['Contents']]
                        if objects:
                            s3_client.delete_objects(
                                Bucket=bucket_name,
                                Delete={'Objects': objects}
                            )
                            print(f"      ✅ Deleted {len(objects)} objects")
                    
                    # Delete any versioned objects
                    response = s3_client.list_object_versions(Bucket=bucket_name)
                    if 'Versions' in response:
                        versions = [
                            {'Key': obj['Key'], 'VersionId': obj['VersionId']} 
                            for obj in response['Versions']
                        ]
                        if versions:
                            s3_client.delete_objects(
                                Bucket=bucket_name,
                                Delete={'Objects': versions}
                            )
                            print(f"      ✅ Deleted {len(versions)} versions")
                            
                except Exception as e:
                    print(f"      ❌ Error emptying {bucket_name}: {e}")
            
            # Now destroy infrastructure with Terraform
            print("   🔥 Destroying Terraform infrastructure...")
            
            os.chdir('terraform/environments/dev')
            
            # Run terraform destroy
            result = subprocess.run([
                'terraform', 'destroy', '-auto-approve'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("      ✅ Terraform destroy completed successfully")
                print("      💀 ALL S3 INFRASTRUCTURE DESTROYED")
                return True
            else:
                print(f"      ❌ Terraform destroy failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error destroying S3 infrastructure: {e}")
            return False
        finally:
            # Return to original directory
            os.chdir('../../..')
    
    def destroy_mongodb_data(self):
        """Destroy all MongoDB collections and data"""
        print("💥 DESTROYING MongoDB Atlas Data...")
        
        try:
            client = MongoClient(self.config['mongodb']['connection_string'])
            db = client[self.config['mongodb']['database_name']]
            
            # Get all collections
            collections = db.list_collection_names()
            print(f"   🔍 Found {len(collections)} collections to destroy")
            
            destroyed_count = 0
            for collection_name in collections:
                try:
                    # Get document count before deletion
                    doc_count = db[collection_name].count_documents({})
                    
                    # DROP THE ENTIRE COLLECTION
                    db[collection_name].drop()
                    
                    print(f"      ✅ DESTROYED collection '{collection_name}' ({doc_count:,} documents)")
                    destroyed_count += 1
                    
                except Exception as e:
                    print(f"      ❌ Error destroying collection {collection_name}: {e}")
            
            print(f"   💀 DESTROYED {destroyed_count} MongoDB collections")
            
            # Verify destruction
            remaining_collections = db.list_collection_names()
            if not remaining_collections:
                print("   ✅ MongoDB database is completely empty")
                return True
            else:
                print(f"   ⚠️  Warning: {len(remaining_collections)} collections still exist")
                return False
                
        except Exception as e:
            print(f"   ❌ Error connecting to MongoDB: {e}")
            return False
        finally:
            client.close()
    
    def destroy_terraform_state(self):
        """Destroy local Terraform state files"""
        print("💥 DESTROYING Terraform State Files...")
        
        try:
            tf_dir = "terraform/environments/dev"
            state_files = [
                'terraform.tfstate',
                'terraform.tfstate.backup',
                '.terraform.lock.hcl'
            ]
            
            destroyed_files = 0
            for state_file in state_files:
                file_path = f"{tf_dir}/{state_file}"
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"   ✅ DESTROYED: {state_file}")
                    destroyed_files += 1
                else:
                    print(f"   ⚠️  Not found: {state_file}")
            
            # Also destroy .terraform directory
            terraform_dir = f"{tf_dir}/.terraform"
            if os.path.exists(terraform_dir):
                import shutil
                shutil.rmtree(terraform_dir)
                print(f"   ✅ DESTROYED: .terraform directory")
                destroyed_files += 1
            
            print(f"   💀 DESTROYED {destroyed_files} Terraform state files")
            return True
            
        except Exception as e:
            print(f"   ❌ Error destroying Terraform state: {e}")
            return False
    
    def simulate_config_corruption(self):
        """Simulate configuration file corruption/loss"""
        print("💥 SIMULATING Configuration File Corruption...")
        
        try:
            # Corrupt the config.yaml file
            corrupted_config = """# CORRUPTED CONFIG FILE - DISASTER SIMULATION
# All original configuration has been lost!
# This simulates accidental deletion or corruption

disaster:
  status: "ACTIVE"
  timestamp: "{}"
  message: "Original configuration destroyed - recovery needed!"
  
# All AWS, MongoDB, and application settings are GONE!
""".format(self.timestamp)
            
            # Backup original first
            import shutil
            shutil.copy2('config.yaml', f'config.yaml.original.{self.timestamp}')
            
            # Overwrite with corrupted version
            with open('config.yaml', 'w') as f:
                f.write(corrupted_config)
            
            print("   ✅ Configuration file corrupted")
            print("   💀 ALL APPLICATION SETTINGS LOST")
            return True
            
        except Exception as e:
            print(f"   ❌ Error corrupting config: {e}")
            return False
    
    def verify_destruction(self):
        """Verify that everything has been properly destroyed"""
        print("🔍 VERIFYING TOTAL DESTRUCTION...")
        
        destruction_complete = True
        
        # Check S3 buckets
        try:
            s3_client = boto3.client('s3', region_name='ca-central-1')  # Default region
            buckets = s3_client.list_buckets()['Buckets']
            arxplorer_buckets = [b for b in buckets if 'arxplorer' in b['Name'].lower()]
            
            if arxplorer_buckets:
                print(f"   ⚠️  WARNING: {len(arxplorer_buckets)} ArXplorer S3 buckets still exist!")
                destruction_complete = False
            else:
                print("   ✅ No ArXplorer S3 buckets found - destruction complete")
                
        except Exception as e:
            print(f"   ✅ Cannot access S3 (expected after destruction): {e}")
        
        # Check MongoDB collections
        try:
            # Try to read corrupted config
            with open('config.yaml', 'r') as f:
                config_content = f.read()
                
            if 'CORRUPTED CONFIG FILE' in config_content:
                print("   ✅ Configuration file successfully corrupted")
            else:
                print("   ⚠️  WARNING: Configuration file not properly corrupted")
                destruction_complete = False
                
        except Exception as e:
            print(f"   ❌ Error checking configuration: {e}")
            destruction_complete = False
        
        # Check Terraform state
        tf_state_file = "terraform/environments/dev/terraform.tfstate"
        if not os.path.exists(tf_state_file):
            print("   ✅ Terraform state files successfully destroyed")
        else:
            print("   ⚠️  WARNING: Terraform state files still exist")
            destruction_complete = False
        
        return destruction_complete

def main():
    """Execute disaster simulation"""
    print("🚨 ArXplorer Disaster Recovery - DISASTER SIMULATION")
    print("=" * 60)
    print("⚠️  WARNING: This will DESTROY all infrastructure and data!")
    print("⚠️  Ensure you have completed the pre-disaster backup first!")
    print("=" * 60)
    
    # Confirmation prompt
    confirmation = input("Type 'DESTROY EVERYTHING' to proceed with disaster simulation: ")
    if confirmation != 'DESTROY EVERYTHING':
        print("❌ Disaster simulation cancelled - incorrect confirmation")
        return
    
    print("🔥 BEGINNING TOTAL DESTRUCTION...")
    print()
    
    disaster = DisasterSimulation()
    
    # Execute destruction sequence
    results = {
        's3_destroyed': disaster.destroy_s3_infrastructure(),
        'mongodb_destroyed': disaster.destroy_mongodb_data(),
        'terraform_state_destroyed': disaster.destroy_terraform_state(),
        'config_corrupted': disaster.simulate_config_corruption()
    }
    
    # Verify total destruction
    destruction_complete = disaster.verify_destruction()
    
    print("\n" + "=" * 60)
    print("💀 DISASTER SIMULATION RESULTS:")
    print(f"   S3 Infrastructure: {'DESTROYED' if results['s3_destroyed'] else 'FAILED'}")
    print(f"   MongoDB Data: {'DESTROYED' if results['mongodb_destroyed'] else 'FAILED'}")
    print(f"   Terraform State: {'DESTROYED' if results['terraform_state_destroyed'] else 'FAILED'}")
    print(f"   Configuration: {'CORRUPTED' if results['config_corrupted'] else 'FAILED'}")
    print(f"   Overall Status: {'TOTAL DESTRUCTION' if destruction_complete else 'PARTIAL DESTRUCTION'}")
    print("=" * 60)
    
    if destruction_complete:
        print("🎬 DISASTER SIMULATION COMPLETE!")
        print("📹 Ready to begin recovery demonstration...")
    else:
        print("⚠️  Disaster simulation incomplete - some components may still exist")

if __name__ == "__main__":
    main()