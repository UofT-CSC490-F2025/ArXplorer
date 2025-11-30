"""
Example usage of AWS IoT MQTT client in ArXplorer
Demonstrates how to integrate MQTT messaging with search functionality.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.aws import AWSIoTMQTTClient, MQTTSearchBridge
from src.core.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleSearchEngine:
    """Mock search engine for demonstration."""
    
    async def search(self, query: str, **filters):
        """Mock search method."""
        logger.info(f"Performing search for: {query}")
        
        # Simulate search results
        results = [
            {
                "title": f"Mock Paper 1 for '{query}'",
                "abstract": f"This is a mock abstract containing information about {query}",
                "authors": ["Author A", "Author B"],
                "arxiv_id": "2023.12345"
            },
            {
                "title": f"Mock Paper 2 for '{query}'", 
                "abstract": f"Another mock paper discussing {query} in detail",
                "authors": ["Author C", "Author D"],
                "arxiv_id": "2023.67890"
            }
        ]
        
        return results


async def example_mqtt_usage():
    """Example of using AWS IoT MQTT client."""
    
    # Initialize MQTT client
    mqtt_client = AWSIoTMQTTClient()
    
    if not mqtt_client.enabled:
        logger.info("AWS IoT MQTT is disabled in configuration")
        return
    
    try:
        # Connect to AWS IoT
        success = await mqtt_client.connect()
        if not success:
            logger.error("Failed to connect to AWS IoT")
            return
        
        # Create search engine and bridge
        search_engine = ExampleSearchEngine()
        bridge = MQTTSearchBridge(mqtt_client, search_engine)
        await bridge.setup()
        
        # Publish a status message
        await mqtt_client.publish_status("online", {"service": "arxplorer"})
        
        # Simulate a search request
        await mqtt_client.publish_search_request(
            query="machine learning transformers",
            filters={"category": "cs.LG", "max_results": 10}
        )
        
        # Keep the connection alive for a while to receive messages
        logger.info("MQTT client running... Press Ctrl+C to stop")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Disconnect
        await mqtt_client.disconnect()


async def example_message_handler(topic: str, message: dict):
    """Example custom message handler."""
    logger.info(f"Received custom message on {topic}: {message}")
    
    # Handle different message types
    if "query" in message:
        query = message["query"]
        logger.info(f"Processing query: {query}")
        
        # Your custom logic here
        # For example, trigger a specific search or analysis


async def example_custom_topics():
    """Example of using custom topics and message handlers."""
    
    mqtt_client = AWSIoTMQTTClient()
    
    if not mqtt_client.enabled:
        logger.info("AWS IoT MQTT is disabled in configuration")
        return
    
    try:
        await mqtt_client.connect()
        
        # Subscribe to custom topics
        await mqtt_client.subscribe("arxplorer/custom/topic", example_message_handler)
        
        # Publish to custom topics
        await mqtt_client.publish(
            "arxplorer/custom/topic",
            {
                "type": "custom_message",
                "data": {"key": "value"},
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
        # Wait for messages
        await asyncio.sleep(10)
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await mqtt_client.disconnect()


if __name__ == "__main__":
    print("AWS IoT MQTT Example Usage")
    print("1. Basic usage with search bridge")
    print("2. Custom topics and handlers")
    
    choice = input("Choose example (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(example_mqtt_usage())
    elif choice == "2":
        asyncio.run(example_custom_topics())
    else:
        print("Invalid choice")