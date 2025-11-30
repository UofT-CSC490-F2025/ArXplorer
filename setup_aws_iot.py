#!/usr/bin/env python3
"""
AWS IoT Setup Script for ArXplorer
Run this script to set up AWS IoT resources and certificates.

Usage:
    python setup_aws_iot.py --thing-name arxplorer-client --region us-east-1
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from src.aws.iot_setup import AWSIoTSetup
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(description='Set up AWS IoT for ArXplorer')
    parser.add_argument('--thing-name', default='arxplorer-client', 
                       help='Name for the IoT thing (default: arxplorer-client)')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region (default: us-east-1)')
    parser.add_argument('--profile', default=None,
                       help='AWS profile name (uses default if not specified)')
    parser.add_argument('--cert-dir', default='certs',
                       help='Directory to store certificates (default: certs)')
    parser.add_argument('--test-connection', action='store_true',
                       help='Test the MQTT connection after setup')
    
    args = parser.parse_args()
    
    print(f"🚀 Setting up AWS IoT for ArXplorer")
    print(f"Thing name: {args.thing_name}")
    print(f"Region: {args.region}")
    print(f"Certificate directory: {args.cert_dir}")
    
    try:
        setup = AWSIoTSetup(region=args.region, profile_name=args.profile)
        
        print("\n📦 Creating AWS IoT resources...")
        result = setup.setup_complete_thing(
            thing_name=args.thing_name,
            cert_dir=args.cert_dir
        )
        
        print("\n✅ AWS IoT setup completed successfully!")
        print(f"Thing name: {result['thing_name']}")
        print(f"Endpoint: {result['endpoint']}")
        print(f"Certificate file: {result['cert_file']}")
        print(f"Private key file: {result['private_key_file']}")
        print(f"CA certificate: {result['ca_cert_file']}")
        
        # Generate config.yaml updates
        config_updates = f"""
# Add these settings to your config.yaml:
aws_iot:
  enabled: true
  region: "{args.region}"
  endpoint: "{result['endpoint']}"
  thing_name: "{args.thing_name}"
  certificates:
    ca_cert: "{result['ca_cert_file']}"
    cert_file: "{result['cert_file']}"
    private_key: "{result['private_key_file']}"
  topics:
    search_requests: "arxplorer/search/requests"
    search_results: "arxplorer/search/results"
    embeddings: "arxplorer/embeddings"
    status: "arxplorer/status"
  qos: 1
  keep_alive: 30
"""
        
        print("\n📝 Configuration updates:")
        print(config_updates)
        
        # Save config updates to file
        with open('aws_iot_config.yaml', 'w') as f:
            f.write(config_updates.strip())
        print("💾 Configuration saved to aws_iot_config.yaml")
        
        # Test connection if requested
        if args.test_connection:
            print("\n🔍 Testing MQTT connection...")
            if setup.test_connection(args.thing_name):
                print("✅ Connection test successful!")
            else:
                print("❌ Connection test failed - check AWS credentials and network")
        
        print(f"\n🎉 Setup complete! Next steps:")
        print(f"1. Update your config.yaml with the settings above")
        print(f"2. Install required packages: pip install awsiotsdk paho-mqtt")
        print(f"3. Test the MQTT client with your ArXplorer application")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())