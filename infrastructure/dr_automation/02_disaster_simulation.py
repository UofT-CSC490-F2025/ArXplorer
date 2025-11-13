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
        # Get the correct path to config.yaml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        config_file = os.path.join(project_root, 'config.yaml')
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def destroy_s3_infrastructure(self):
        """Destroy all S3 buckets and data using Terraform"""
        print("DESTROYING AWS S3 Infrastructure...")
        
        try:
            # First, empty all buckets (Terraform can't destroy non-empty buckets)
            s3_client = boto3.client('s3', region_name=self.config['aws']['region'])
            
            for bucket_type, bucket_name in self.config['aws']['buckets'].items():
                try:
                    print(f"   Emptying bucket: {bucket_name}")
                    
                    # List and delete all objects
                    response = s3_client.list_objects_v2(Bucket=bucket_name)
                    
                    if 'Contents' in response:
                        objects = [{'Key': obj['Key']} for obj in response['Contents']]
                        if objects:
                            s3_client.delete_objects(
                                Bucket=bucket_name,
                                Delete={'Objects': objects}
                            )
                            print(f"      SUCCESS: Deleted {len(objects)} objects")
                    
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
                            print(f"      SUCCESS: Deleted {len(versions)} versions")
                            
                except Exception as e:
                    print(f"      ERROR: Error emptying {bucket_name}: {e}")
            
            # Now destroy infrastructure with Terraform
            print("   Destroying Terraform infrastructure...")
            
            # Get current directory and terraform directory
            original_dir = os.getcwd()
            tf_dir = os.path.join(original_dir, 'infrastructure', 'terraform', 'environments', 'dev')
            
            if os.path.exists(tf_dir):
                os.chdir(tf_dir)
                
                # Check if terraform is available - try different locations
                terraform_paths = [
                    'terraform',
                    'terraform.exe',
                    r'C:\Program Files\terraform\terraform.exe',
                    r'C:\Users\Admin\AppData\Local\Microsoft\WinGet\Links\terraform.exe'
                ]
                
                terraform_cmd = None
                for path in terraform_paths:
                    if path in ['terraform', 'terraform.exe']:
                        if os.system(f'where {path} >nul 2>&1') == 0:
                            terraform_cmd = path
                            break
                    else:
                        if os.path.exists(path):
                            terraform_cmd = path
                            break
                
                if not terraform_cmd:
                    print("      WARNING: Terraform not found, skipping terraform destroy")
                    os.chdir(original_dir)
                    # Still empty buckets manually since that's the main goal
                    return True
                
                # Run terraform destroy
                result = subprocess.run([
                    terraform_cmd, 'destroy', '-auto-approve'
                ], capture_output=True, text=True)
                
                os.chdir(original_dir)
            else:
                print(f"      WARNING: Terraform directory not found: {tf_dir}")
                result = None
            
            if result and result.returncode == 0:
                print("      SUCCESS: Terraform destroy completed successfully")
                print("      ALL S3 INFRASTRUCTURE DESTROYED")
                return True
            elif result:
                print(f"      ERROR: Terraform destroy failed: {result.stderr}")
                return False
            else:
                # If terraform not available, still consider S3 buckets emptied as partial success
                print("      WARNING: Terraform destroy skipped, but S3 buckets emptied")
                return True
                
        except Exception as e:
            print(f"   ERROR: Error destroying S3 infrastructure: {e}")
            return False
        finally:
            # Return to original directory
            os.chdir('../../..')
    
    def destroy_mongodb_data(self):
        """Destroy all MongoDB collections and data"""
        print("DESTROYING MongoDB Atlas Data...")
        
        client = None
        try:
            client = MongoClient(self.config['mongodb']['connection_string'])
            db = client[self.config['mongodb']['database_name']]
            
            # Get all collections
            collections = db.list_collection_names()
            print(f"   Found {len(collections)} collections to destroy")
            
            destroyed_count = 0
            for collection_name in collections:
                try:
                    # Get document count before deletion
                    doc_count = db[collection_name].count_documents({})
                    
                    # DROP THE ENTIRE COLLECTION
                    db[collection_name].drop()
                    
                    print(f"      SUCCESS: DESTROYED collection '{collection_name}' ({doc_count:,} documents)")
                    destroyed_count += 1
                    
                except Exception as e:
                    print(f"      ERROR: Error destroying collection {collection_name}: {e}")
            
            print(f"   DESTROYED {destroyed_count} MongoDB collections")
            
            # Verify destruction
            remaining_collections = db.list_collection_names()
            if not remaining_collections:
                print("   SUCCESS: MongoDB database is completely empty")
                return True
            else:
                print(f"   WARNING: {len(remaining_collections)} collections still exist")
                return False
                
        except Exception as e:
            print(f"   ERROR: Error connecting to MongoDB: {e}")
            return False
        finally:
            if client:
                client.close()
    
    def destroy_terraform_state(self):
        """Destroy local Terraform state files"""
        print("DESTROYING Terraform State Files...")
        
        try:
            tf_dir = "infrastructure/terraform/environments/dev"
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
                    print(f"   SUCCESS: DESTROYED: {state_file}")
                    destroyed_files += 1
                else:
                    print(f"   WARNING: Not found: {state_file}")
            
            # Also destroy .terraform directory
            terraform_dir = f"{tf_dir}/.terraform"
            if os.path.exists(terraform_dir):
                import shutil
                shutil.rmtree(terraform_dir)
                print(f"   SUCCESS: DESTROYED: .terraform directory")
                destroyed_files += 1
            
            print(f"   DESTROYED {destroyed_files} Terraform state files")
            return True
            
        except Exception as e:
            print(f"   ERROR: Error destroying Terraform state: {e}")
            return False
    
    def simulate_config_corruption(self, destroy_completely=False):
        """Simulate configuration file corruption/loss"""
        if destroy_completely:
            print("DESTROYING Configuration File...")
        else:
            print("SIMULATING Configuration File Corruption...")
        
        try:
            # Get the root project directory
            # Since script is in infrastructure/disaster_recovery, go up two levels
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            
            config_file = os.path.join(project_root, 'config.yaml')
            
            if not os.path.exists(config_file):
                print(f"   WARNING: config.yaml not found at {config_file}")
                return False
            
            # Backup original first
            import shutil
            shutil.copy2(config_file, f'{config_file}.original.{self.timestamp}')
            
            if destroy_completely:
                # Actually delete the configuration file
                os.remove(config_file)
                print("   SUCCESS: Configuration file COMPLETELY DESTROYED")
                print("   ALL APPLICATION SETTINGS DELETED")
                return True
            else:
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
                
                # Overwrite with corrupted version
                with open(config_file, 'w') as f:
                    f.write(corrupted_config)
                
                print("   SUCCESS: Configuration file corrupted")
                print("   ALL APPLICATION SETTINGS LOST")
                return True
            
        except Exception as e:
            print(f"   ERROR: Error corrupting config: {e}")
            return False
    
    def verify_destruction(self):
        """Verify that everything has been properly destroyed"""
        print("VERIFYING TOTAL DESTRUCTION...")
        
        destruction_complete = True
        
        # Check S3 buckets
        try:
            s3_client = boto3.client('s3', region_name='ca-central-1')  # Default region
            buckets = s3_client.list_buckets()['Buckets']
            arxplorer_buckets = [b for b in buckets if 'arxplorer' in b['Name'].lower()]
            
            if arxplorer_buckets:
                print(f"   WARNING: {len(arxplorer_buckets)} ArXplorer S3 buckets still exist!")
                destruction_complete = False
            else:
                print("   SUCCESS: No ArXplorer S3 buckets found - destruction complete")
                
        except Exception as e:
            print(f"   SUCCESS: Cannot access S3 (expected after destruction): {e}")
        
        # Check MongoDB collections
        try:
            # Get the correct path to config.yaml
            # Since script is in infrastructure/disaster_recovery, go up two levels
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            
            config_file = os.path.join(project_root, 'config.yaml')
            
            # Try to read corrupted config
            with open(config_file, 'r') as f:
                config_content = f.read()
                
            if 'CORRUPTED CONFIG FILE' in config_content:
                print("   SUCCESS: Configuration file successfully corrupted")
            else:
                print("   WARNING: Configuration file not properly corrupted")
                destruction_complete = False
                
        except Exception as e:
            print(f"   ERROR: Error checking configuration: {e}")
            destruction_complete = False
        
        # Check Terraform state
        tf_state_file = "infrastructure/terraform/environments/dev/terraform.tfstate"
        if not os.path.exists(tf_state_file):
            print("   SUCCESS: Terraform state files successfully destroyed")
        else:
            print("   WARNING: Terraform state files still exist")
            destruction_complete = False
        
        return destruction_complete

def main():
    """Execute disaster simulation"""
    print("ArXplorer Disaster Recovery - DISASTER SIMULATION")
    print("=" * 60)
    print("WARNING: This will DESTROY all infrastructure and data!")
    print("WARNING: Ensure you have completed the pre-disaster backup first!")
    print("=" * 60)
    
    # Confirmation prompt
    confirmation = input("Type 'DESTROY EVERYTHING' to proceed with disaster simulation: ")
    if confirmation != 'DESTROY EVERYTHING':
        print("ERROR: Disaster simulation cancelled - incorrect confirmation")
        return
    
    # Ask about destruction mode
    print("\nDestruction Mode Options:")
    print("1. CORRUPT configuration (simulate corruption - file remains but damaged)")
    print("2. DESTROY configuration (completely delete file - more realistic)")
    mode = input("Choose destruction mode (1 or 2): ").strip()
    
    destroy_config_completely = (mode == '2')
    
    print("BEGINNING TOTAL DESTRUCTION...")
    print()
    
    disaster = DisasterSimulation()
    
    # Execute destruction sequence
    results = {
        's3_destroyed': disaster.destroy_s3_infrastructure(),
        'mongodb_destroyed': disaster.destroy_mongodb_data(),
        'terraform_state_destroyed': disaster.destroy_terraform_state(),
        'config_corrupted': disaster.simulate_config_corruption(destroy_completely=destroy_config_completely)
    }
    
    # Verify total destruction
    destruction_complete = disaster.verify_destruction()
    
    print("\n" + "=" * 60)
    print("DISASTER SIMULATION RESULTS:")
    print(f"   S3 Infrastructure: {'DESTROYED' if results['s3_destroyed'] else 'FAILED'}")
    print(f"   MongoDB Data: {'DESTROYED' if results['mongodb_destroyed'] else 'FAILED'}")
    print(f"   Terraform State: {'DESTROYED' if results['terraform_state_destroyed'] else 'FAILED'}")
    print(f"   Configuration: {'CORRUPTED' if results['config_corrupted'] else 'FAILED'}")
    print(f"   Overall Status: {'TOTAL DESTRUCTION' if destruction_complete else 'PARTIAL DESTRUCTION'}")
    print("=" * 60)
    
    if destruction_complete:
        print("DISASTER SIMULATION COMPLETE!")
        print("📹 Ready to begin recovery demonstration...")
    else:
        print("WARNING: Disaster simulation incomplete - some components may still exist")

if __name__ == "__main__":
    main()