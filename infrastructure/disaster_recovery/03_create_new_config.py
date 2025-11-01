#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Configuration Recreation Script
Creates a new configuration file when original is completely lost
"""

import os
import yaml
from datetime import datetime

def create_new_config():
    """Create a new configuration file from scratch"""
    print("ArXplorer Configuration Recovery - Creating New Config")
    print("=" * 60)
    
    # Basic configuration template
    new_config = {
        'arxiv': {
            'api_url': 'http://export.arxiv.org/api/query',
            'categories': ['cs.AI', 'cs.CL', 'cs.LG', 'cs.IR', 'stat.ML', 'cs.CV'],
            'rate_limit_delay': 1.0
        },
        'processing': {
            'batch_size': 100,
            'max_workers': 4,
            'retry_attempts': 3,
            'timeout': 300
        },
        'embedding': {
            'model_name': 'allenai/scibert_scivocab_uncased',
            'cache_dir': './models/cache',
            'max_length': 512,
            'batch_size': 32
        },
        'mongodb': {
            'connection_string': 'PLACEHOLDER_MONGODB_CONNECTION',
            'database_name': 'arxplorer',
            'collections': {
                'papers': 'papers',
                'processed_papers': 'processed_papers', 
                'embeddings': 'embeddings',
                'search_cache': 'search_cache'
            }
        },
        'aws': {
            'region': 'ca-central-1',
            's3': {
                'raw_data_bucket': 'PLACEHOLDER_S3_BUCKET',
                'processed_bucket': 'PLACEHOLDER_S3_BUCKET',
                'embeddings_bucket': 'PLACEHOLDER_S3_BUCKET'
            }
        },
        'server': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False,
            'reload': False
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/arxplorer.log'
        },
        'recovery': {
            'created_at': datetime.now().isoformat(),
            'created_by': 'disaster_recovery_script',
            'note': 'Configuration recreated after disaster recovery'
        }
    }
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    config_file = os.path.join(project_root, 'config.yaml')
    
    # Backup existing config if it exists
    if os.path.exists(config_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{config_file}.backup.{timestamp}"
        os.rename(config_file, backup_file)
        print(f"   Backed up existing config to: {backup_file}")
    
    # Write new configuration
    with open(config_file, 'w') as f:
        f.write("# ArXplorer Configuration - Recreated after Disaster Recovery\n")
        f.write(f"# Created: {datetime.now().isoformat()}\n")
        f.write("# NOTE: Please update MongoDB and AWS credentials below\n\n")
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"SUCCESS: New configuration created at: {config_file}")
    print("\nIMPORTANT: You need to update the following placeholders:")
    print("  - PLACEHOLDER_MONGODB_CONNECTION: Update with your MongoDB Atlas connection string")
    print("  - PLACEHOLDER_S3_BUCKET: Update with your actual S3 bucket names")
    print("\nConfiguration has been recreated with standard defaults.")
    
    return config_file

if __name__ == "__main__":
    create_new_config()