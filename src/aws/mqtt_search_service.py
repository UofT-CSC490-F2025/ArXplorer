"""
MQTT-enabled Search Service for ArXplorer
Integrates AWS IoT MQTT messaging with the existing search functionality.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from ..search import HybridSearchEngine, SearchAPI, SearchMode
from ..aws import AWSIoTMQTTClient, MQTTSearchBridge
from ..core.config import get_config

logger = logging.getLogger(__name__)


class MQTTSearchService:
    """
    MQTT-enabled search service that combines ArXplorer's search capabilities
    with AWS IoT messaging for distributed search requests.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the MQTT search service."""
        self.config = config or get_config()
        self.search_api = None
        self.mqtt_client = None
        self.bridge = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize search engine and MQTT components."""
        try:
            # Initialize search API
            logger.info("Initializing search engine...")
            self.search_api = SearchAPI(self.config)
            await self.search_api.initialize()
            
            # Initialize MQTT client if enabled
            mqtt_config = self.config.get('aws_iot', {})
            if mqtt_config.get('enabled', False):
                logger.info("Initializing AWS IoT MQTT client...")
                self.mqtt_client = AWSIoTMQTTClient(mqtt_config)
                
                # Create search bridge
                self.bridge = MQTTSearchBridge(self.mqtt_client, self)
                
                logger.info("MQTT search service initialized successfully")
            else:
                logger.info("MQTT disabled - running in local mode only")
                
        except Exception as e:
            logger.error(f"Failed to initialize MQTT search service: {e}")
            raise
    
    async def start(self):
        """Start the MQTT search service."""
        if self.is_running:
            logger.warning("Service is already running")
            return
            
        try:
            # Connect MQTT if available
            if self.mqtt_client:
                success = await self.mqtt_client.connect()
                if success:
                    await self.bridge.setup()
                    await self.mqtt_client.publish_status(
                        "online", 
                        {"service": "arxplorer_search", "capabilities": ["hybrid_search", "llm_judge"]}
                    )
                    logger.info("MQTT search service started and online")
                else:
                    logger.warning("MQTT connection failed - running in local mode")
            
            self.is_running = True
            logger.info("Search service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MQTT search service: {e}")
            raise
    
    async def stop(self):
        """Stop the MQTT search service."""
        if not self.is_running:
            return
            
        try:
            if self.mqtt_client:
                await self.mqtt_client.publish_status(
                    "offline", 
                    {"service": "arxplorer_search", "reason": "shutdown"}
                )
                await self.mqtt_client.disconnect()
                
            self.is_running = False
            logger.info("MQTT search service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MQTT search service: {e}")
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform search using the underlying search API.
        This method is called by the MQTT bridge and can also be used directly.
        """
        try:
            # Extract search parameters
            mode_str = kwargs.get('mode', 'hybrid')
            max_results = kwargs.get('max_results', 20)
            categories = kwargs.get('categories', None)
            
            # Convert mode string to SearchMode enum
            if isinstance(mode_str, str):
                mode_map = {
                    'hybrid': SearchMode.HYBRID,
                    'semantic': SearchMode.SEMANTIC, 
                    'keyword': SearchMode.KEYWORD
                }
                mode = mode_map.get(mode_str.lower(), SearchMode.HYBRID)
            else:
                mode = mode_str
            
            # Perform search
            logger.info(f"Performing {mode.value} search for: '{query}'")
            results = await self.search_api.search(
                query=query,
                mode=mode,
                max_results=max_results,
                categories=categories
            )
            
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                if hasattr(result, 'to_dict'):
                    serializable_results.append(result.to_dict())
                elif isinstance(result, dict):
                    serializable_results.append(result)
                else:
                    # Convert SearchResult to dict
                    serializable_results.append({
                        'title': getattr(result, 'title', ''),
                        'abstract': getattr(result, 'abstract', ''),
                        'authors': getattr(result, 'authors', []),
                        'arxiv_id': getattr(result, 'arxiv_id', ''),
                        'categories': getattr(result, 'categories', []),
                        'score': getattr(result, 'score', 0.0),
                        'published': getattr(result, 'published', None),
                        'url': getattr(result, 'url', '')
                    })
            
            logger.info(f"Found {len(serializable_results)} results")
            return serializable_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    async def handle_search_request_mqtt(self, topic: str, message: Dict[str, Any]):
        """Handle search requests from MQTT."""
        try:
            query = message.get('query', '')
            client_id = message.get('client_id', 'unknown')
            request_id = message.get('request_id', 'unknown')
            
            if not query:
                logger.warning(f"Empty query from client {client_id}")
                return
            
            logger.info(f"Processing MQTT search request from {client_id}: {query}")
            
            # Extract search parameters
            filters = message.get('filters', {})
            
            # Perform search
            results = await self.search(query, **filters)
            
            # Publish results back via MQTT
            await self.mqtt_client.publish_search_results(
                query=query,
                results=results,
                metadata={
                    'processed_by': self.mqtt_client.thing_name,
                    'request_id': request_id,
                    'client_id': client_id,
                    'processing_time_ms': message.get('processing_time_ms', 0)
                }
            )
            
            logger.info(f"Published {len(results)} results for MQTT request: {query}")
            
        except Exception as e:
            logger.error(f"Error handling MQTT search request: {e}")
            # Publish error status
            if self.mqtt_client:
                await self.mqtt_client.publish_status(
                    status='error',
                    details={
                        'error': str(e),
                        'query': message.get('query', ''),
                        'client_id': message.get('client_id', 'unknown')
                    }
                )
    
    async def run_forever(self):
        """Run the service indefinitely."""
        try:
            await self.start()
            logger.info("MQTT search service is running. Press Ctrl+C to stop.")
            
            while self.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Service error: {e}")
        finally:
            await self.stop()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


async def main():
    """Main function to run the MQTT search service."""
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        service = MQTTSearchService()
        await service.run_forever()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())