#!/usr/bin/env python3
"""
Quick Start Script for AWS IoT MQTT Integration
Validates setup and demonstrates basic functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import boto3
        import yaml
        logger.info("✅ Core dependencies available")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install with: pip install boto3 pyyaml")
        return False
    
    # Check optional MQTT dependencies
    mqtt_available = False
    try:
        import awsiot
        import paho.mqtt.client
        mqtt_available = True
        logger.info("✅ MQTT dependencies available")
    except ImportError:
        logger.warning("⚠️ MQTT dependencies not available")
        logger.warning("Install with: pip install awsiotsdk paho-mqtt")
    
    return mqtt_available

async def check_configuration():
    """Check if AWS IoT configuration is available."""
    config_file = Path("config.yaml")
    if not config_file.exists():
        logger.error("❌ config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        aws_iot_config = config.get('aws_iot', {})
        if not aws_iot_config.get('enabled', False):
            logger.info("ℹ️ AWS IoT is disabled in configuration")
            return False
        
        required_fields = ['endpoint', 'thing_name', 'region']
        for field in required_fields:
            if not aws_iot_config.get(field):
                logger.error(f"❌ Missing AWS IoT config field: {field}")
                return False
        
        logger.info("✅ AWS IoT configuration looks good")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading configuration: {e}")
        return False

async def check_certificates():
    """Check if certificates are available."""
    cert_dir = Path("certs")
    required_certs = [
        "AmazonRootCA1.pem",
        "device-certificate.pem.crt", 
        "device-private.pem.key"
    ]
    
    missing_certs = []
    for cert in required_certs:
        cert_path = cert_dir / cert
        if not cert_path.exists():
            missing_certs.append(str(cert_path))
    
    if missing_certs:
        logger.error("❌ Missing certificates:")
        for cert in missing_certs:
            logger.error(f"   - {cert}")
        logger.error("Run: python setup_aws_iot.py to generate certificates")
        return False
    
    logger.info("✅ All required certificates found")
    return True

async def test_aws_connection():
    """Test basic AWS connection."""
    try:
        import boto3
        
        # Try to create IoT client
        iot_client = boto3.client('iot', region_name='us-east-1')
        
        # Test connection with a simple operation
        response = iot_client.describe_endpoint(endpointType='iot:Data-ATS')
        endpoint = response['endpointAddress']
        
        logger.info(f"✅ AWS IoT endpoint reachable: {endpoint}")
        return True
        
    except Exception as e:
        logger.error(f"❌ AWS connection failed: {e}")
        logger.error("Check your AWS credentials with: aws iot describe-endpoint --endpoint-type iot:Data-ATS")
        return False

async def test_mqtt_connection():
    """Test MQTT connection to AWS IoT."""
    try:
        # Add src to path for imports
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        from src.aws import AWSIoTMQTTClient
        
        client = AWSIoTMQTTClient()
        if not client.enabled:
            logger.info("ℹ️ MQTT client is disabled")
            return True
        
        logger.info("🔌 Testing MQTT connection...")
        success = await client.connect()
        
        if success:
            logger.info("✅ MQTT connection successful")
            
            # Test publishing
            await client.publish_status("test", {"message": "Quick start test"})
            logger.info("✅ MQTT publish successful")
            
            await client.disconnect()
            return True
        else:
            logger.error("❌ MQTT connection failed")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure src/aws modules are available")
        return False
    except Exception as e:
        logger.error(f"❌ MQTT test failed: {e}")
        return False

async def run_quick_demo():
    """Run a quick demo of the MQTT search service."""
    try:
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        from src.aws.mqtt_search_service import MQTTSearchService
        
        logger.info("🚀 Starting quick demo...")
        
        service = MQTTSearchService()
        await service.initialize()
        
        if service.mqtt_client and service.mqtt_client.enabled:
            await service.start()
            
            # Send a test search request
            await service.mqtt_client.publish_search_request(
                query="test query",
                filters={"category": "cs.AI", "max_results": 5}
            )
            
            logger.info("📡 Published test search request")
            logger.info("💤 Waiting 5 seconds for any responses...")
            await asyncio.sleep(5)
            
            await service.stop()
            logger.info("✅ Demo completed successfully")
        else:
            logger.info("ℹ️ MQTT disabled - demo skipped")
            
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")

async def main():
    """Main quick start function."""
    print("🚀 ArXplorer AWS IoT MQTT Quick Start")
    print("=" * 50)
    
    # Check dependencies
    mqtt_deps = await check_dependencies()
    
    # Check configuration
    config_ok = await check_configuration()
    
    # Check certificates (only if config is OK)
    certs_ok = config_ok and await check_certificates()
    
    # Test AWS connection
    aws_ok = await test_aws_connection()
    
    # Test MQTT (only if all other checks pass)
    mqtt_ok = True
    if mqtt_deps and config_ok and certs_ok and aws_ok:
        mqtt_ok = await test_mqtt_connection()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print(f"Dependencies:     {'✅' if mqtt_deps else '⚠️'}")
    print(f"Configuration:    {'✅' if config_ok else '❌'}")
    print(f"Certificates:     {'✅' if certs_ok else '❌'}")
    print(f"AWS Connection:   {'✅' if aws_ok else '❌'}")
    print(f"MQTT Connection:  {'✅' if mqtt_ok else '❌'}")
    
    if all([mqtt_deps, config_ok, certs_ok, aws_ok, mqtt_ok]):
        print("\n🎉 All checks passed! AWS IoT MQTT is ready to use.")
        
        # Ask if user wants to run demo
        response = input("\nRun quick demo? (y/N): ").strip().lower()
        if response == 'y':
            await run_quick_demo()
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        if not config_ok or not certs_ok:
            print("💡 Try running: python setup_aws_iot.py")

if __name__ == "__main__":
    asyncio.run(main())