"""
AWS IoT Setup and Configuration Utilities
Helper functions for setting up AWS IoT resources and certificates.
"""

import json
import logging
import boto3
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import requests


logger = logging.getLogger(__name__)


class AWSIoTSetup:
    """Helper class for setting up AWS IoT resources."""
    
    def __init__(self, region: str = 'us-east-1', profile_name: Optional[str] = None):
        """Initialize AWS IoT setup helper."""
        self.region = region
        self.session = boto3.Session(profile_name=profile_name)
        self.iot_client = self.session.client('iot', region_name=region)
        self.iot_data_client = self.session.client('iot-data', region_name=region)
        
    def get_iot_endpoint(self) -> str:
        """Get the AWS IoT endpoint URL."""
        try:
            response = self.iot_client.describe_endpoint(endpointType='iot:Data-ATS')
            endpoint = response['endpointAddress']
            logger.info(f"AWS IoT endpoint: {endpoint}")
            return endpoint
        except Exception as e:
            logger.error(f"Failed to get IoT endpoint: {e}")
            raise
    
    def create_thing(self, thing_name: str, thing_type: Optional[str] = None) -> Dict[str, Any]:
        """Create an IoT thing."""
        try:
            params = {'thingName': thing_name}
            if thing_type:
                params['thingTypeName'] = thing_type
                
            response = self.iot_client.create_thing(**params)
            logger.info(f"Created IoT thing: {thing_name}")
            return response
        except self.iot_client.exceptions.ResourceAlreadyExistsException:
            logger.info(f"IoT thing {thing_name} already exists")
            return self.iot_client.describe_thing(thingName=thing_name)
        except Exception as e:
            logger.error(f"Failed to create IoT thing {thing_name}: {e}")
            raise
    
    def create_policy(self, policy_name: str, thing_name: str) -> Dict[str, Any]:
        """Create an IoT policy for the thing."""
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "iot:Connect"
                    ],
                    "Resource": f"arn:aws:iot:{self.region}:*:client/{thing_name}"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "iot:Subscribe"
                    ],
                    "Resource": [
                        f"arn:aws:iot:{self.region}:*:topicfilter/arxplorer/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "iot:Publish",
                        "iot:Receive"
                    ],
                    "Resource": [
                        f"arn:aws:iot:{self.region}:*:topic/arxplorer/*"
                    ]
                }
            ]
        }
        
        try:
            response = self.iot_client.create_policy(
                policyName=policy_name,
                policyDocument=json.dumps(policy_document)
            )
            logger.info(f"Created IoT policy: {policy_name}")
            return response
        except self.iot_client.exceptions.ResourceAlreadyExistsException:
            logger.info(f"IoT policy {policy_name} already exists")
            return {'policyName': policy_name}
        except Exception as e:
            logger.error(f"Failed to create IoT policy {policy_name}: {e}")
            raise
    
    def create_keys_and_certificate(self) -> Tuple[str, str, str, str]:
        """Create keys and certificate for the IoT thing."""
        try:
            response = self.iot_client.create_keys_and_certificate(setAsActive=True)
            
            certificate_arn = response['certificateArn']
            certificate_id = response['certificateId']
            certificate_pem = response['certificatePem']
            private_key = response['keyPair']['PrivateKey']
            public_key = response['keyPair']['PublicKey']
            
            logger.info(f"Created certificate: {certificate_id}")
            
            return certificate_arn, certificate_pem, private_key, public_key
            
        except Exception as e:
            logger.error(f"Failed to create keys and certificate: {e}")
            raise
    
    def attach_policy_to_certificate(self, policy_name: str, certificate_arn: str):
        """Attach policy to certificate."""
        try:
            self.iot_client.attach_policy(
                policyName=policy_name,
                target=certificate_arn
            )
            logger.info(f"Attached policy {policy_name} to certificate {certificate_arn}")
        except Exception as e:
            logger.error(f"Failed to attach policy to certificate: {e}")
            raise
    
    def attach_thing_principal(self, thing_name: str, certificate_arn: str):
        """Attach certificate to thing."""
        try:
            self.iot_client.attach_thing_principal(
                thingName=thing_name,
                principal=certificate_arn
            )
            logger.info(f"Attached certificate to thing {thing_name}")
        except Exception as e:
            logger.error(f"Failed to attach certificate to thing: {e}")
            raise
    
    def download_root_ca(self, cert_path: str = "certs/AmazonRootCA1.pem") -> bool:
        """Download Amazon Root CA certificate."""
        try:
            # Create certs directory if it doesn't exist
            Path(cert_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download Amazon Root CA 1
            ca_url = "https://www.amazontrust.com/repository/AmazonRootCA1.pem"
            response = requests.get(ca_url)
            response.raise_for_status()
            
            with open(cert_path, 'w') as f:
                f.write(response.text)
                
            logger.info(f"Downloaded root CA certificate to {cert_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download root CA certificate: {e}")
            return False
    
    def setup_complete_thing(
        self, 
        thing_name: str,
        cert_dir: str = "certs",
        thing_type: Optional[str] = None
    ) -> Dict[str, str]:
        """Complete setup of an IoT thing with certificates and policies."""
        try:
            # Create certificates directory
            cert_path = Path(cert_dir)
            cert_path.mkdir(parents=True, exist_ok=True)
            
            # Create thing
            thing_response = self.create_thing(thing_name, thing_type)
            
            # Create policy
            policy_name = f"{thing_name}-policy"
            self.create_policy(policy_name, thing_name)
            
            # Create keys and certificate
            cert_arn, cert_pem, private_key, public_key = self.create_keys_and_certificate()
            
            # Attach policy to certificate
            self.attach_policy_to_certificate(policy_name, cert_arn)
            
            # Attach certificate to thing
            self.attach_thing_principal(thing_name, cert_arn)
            
            # Save certificates to files
            cert_file = cert_path / "device-certificate.pem.crt"
            private_key_file = cert_path / "device-private.pem.key"
            public_key_file = cert_path / "device-public.pem.key"
            
            with open(cert_file, 'w') as f:
                f.write(cert_pem)
            
            with open(private_key_file, 'w') as f:
                f.write(private_key)
                
            with open(public_key_file, 'w') as f:
                f.write(public_key)
            
            # Download root CA
            self.download_root_ca(str(cert_path / "AmazonRootCA1.pem"))
            
            # Get IoT endpoint
            endpoint = self.get_iot_endpoint()
            
            logger.info(f"Complete setup for thing {thing_name} finished successfully")
            
            return {
                'thing_name': thing_name,
                'endpoint': endpoint,
                'certificate_arn': cert_arn,
                'policy_name': policy_name,
                'cert_file': str(cert_file),
                'private_key_file': str(private_key_file),
                'public_key_file': str(public_key_file),
                'ca_cert_file': str(cert_path / "AmazonRootCA1.pem")
            }
            
        except Exception as e:
            logger.error(f"Failed to setup IoT thing {thing_name}: {e}")
            raise
    
    def test_connection(self, thing_name: str, topic: str = "test/topic") -> bool:
        """Test MQTT connection by publishing a test message."""
        try:
            # Get endpoint if not set
            if not hasattr(self, '_endpoint'):
                self._endpoint = self.get_iot_endpoint()
                
            # Update the IoT data client with the endpoint
            self.iot_data_client = self.session.client(
                'iot-data', 
                region_name=self.region,
                endpoint_url=f'https://{self._endpoint}'
            )
            
            test_message = {
                'message': 'Test message from ArXplorer',
                'timestamp': str(boto3.Session().get_partition_for_region(self.region)),
                'thing_name': thing_name
            }
            
            response = self.iot_data_client.publish(
                topic=topic,
                qos=1,
                payload=json.dumps(test_message)
            )
            
            logger.info(f"Successfully published test message to {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to test connection: {e}")
            return False


def generate_setup_script(thing_name: str, region: str = 'us-east-1') -> str:
    """Generate a setup script for AWS IoT configuration."""
    script = f"""#!/usr/bin/env python3
\"\"\"
AWS IoT Setup Script for ArXplorer
Run this script to set up AWS IoT resources for {thing_name}
\"\"\"

import asyncio
from src.aws.iot_setup import AWSIoTSetup

async def main():
    setup = AWSIoTSetup(region='{region}')
    
    try:
        result = setup.setup_complete_thing('{thing_name}')
        
        print("✅ AWS IoT setup completed successfully!")
        print(f"Thing name: {{result['thing_name']}}")
        print(f"Endpoint: {{result['endpoint']}}")
        print(f"Certificate file: {{result['cert_file']}}")
        print(f"Private key file: {{result['private_key_file']}}")
        print(f"CA certificate: {{result['ca_cert_file']}}")
        
        # Update config.yaml with the new values
        config_updates = f'''
# Update your config.yaml with these values:
aws_iot:
  enabled: true
  region: "{region}"
  endpoint: "{{result['endpoint']}}"
  thing_name: "{thing_name}"
  certificates:
    ca_cert: "{{result['ca_cert_file']}}"
    cert_file: "{{result['cert_file']}}"
    private_key: "{{result['private_key_file']}}"
'''
        print(config_updates)
        
        # Test connection
        if setup.test_connection(thing_name):
            print("✅ Connection test successful!")
        else:
            print("❌ Connection test failed")
            
    except Exception as e:
        print(f"❌ Setup failed: {{e}}")

if __name__ == "__main__":
    asyncio.run(main())
"""
    return script