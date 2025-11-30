"""
AWS IoT MQTT Client for ArXplorer
Handles MQTT communication with AWS IoT Core for search requests and results.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import ssl
from concurrent.futures import ThreadPoolExecutor

try:
    from awsiot import mqtt_connection_builder
    from awscrt import mqtt
    AWS_IOT_SDK_AVAILABLE = True
except ImportError:
    AWS_IOT_SDK_AVAILABLE = False
    logging.warning("AWS IoT SDK not available. Install with: pip install awsiotsdk")

try:
    import paho.mqtt.client as mqtt_client
    PAHO_MQTT_AVAILABLE = True
except ImportError:
    PAHO_MQTT_AVAILABLE = False
    logging.warning("Paho MQTT not available. Install with: pip install paho-mqtt")

from ..core.config import get_config


logger = logging.getLogger(__name__)


class AWSIoTMQTTClient:
    """AWS IoT MQTT client for handling search requests and responses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AWS IoT MQTT client."""
        self.config = config or get_config().get('aws_iot', {})
        self.enabled = self.config.get('enabled', False)
        
        if not self.enabled:
            logger.info("AWS IoT MQTT client is disabled")
            return
            
        self.endpoint = self.config.get('endpoint', '')
        self.thing_name = self.config.get('thing_name', 'arxplorer-client')
        self.region = self.config.get('region', 'us-east-1')
        self.topics = self.config.get('topics', {})
        self.qos = self.config.get('qos', 1)
        self.keep_alive = self.config.get('keep_alive', 30)
        
        # Certificate paths
        certs = self.config.get('certificates', {})
        self.ca_cert = certs.get('ca_cert', 'certs/AmazonRootCA1.pem')
        self.cert_file = certs.get('cert_file', 'certs/device-certificate.pem.crt')
        self.private_key = certs.get('private_key', 'certs/device-private.pem.key')
        
        self.connection = None
        self.is_connected = False
        self.message_handlers = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the MQTT configuration."""
        if not self.enabled:
            return
            
        if not self.endpoint:
            raise ValueError("AWS IoT endpoint is required")
            
        # Check certificate files exist
        cert_paths = [self.ca_cert, self.cert_file, self.private_key]
        for cert_path in cert_paths:
            if not Path(cert_path).exists():
                logger.warning(f"Certificate file not found: {cert_path}")
    
    async def connect(self) -> bool:
        """Connect to AWS IoT Core."""
        if not self.enabled:
            logger.info("MQTT client is disabled")
            return False
            
        if not AWS_IOT_SDK_AVAILABLE:
            logger.error("AWS IoT SDK not available")
            return False
            
        try:
            # Build MQTT connection
            self.connection = mqtt_connection_builder.mtls_from_path(
                endpoint=self.endpoint,
                cert_filepath=self.cert_file,
                pri_key_filepath=self.private_key,
                ca_filepath=self.ca_cert,
                client_id=self.thing_name,
                clean_session=False,
                keep_alive_secs=self.keep_alive
            )
            
            # Set connection callbacks
            self.connection.on_connection_interrupted = self._on_connection_interrupted
            self.connection.on_connection_resumed = self._on_connection_resumed
            
            # Connect
            logger.info(f"Connecting to AWS IoT Core at {self.endpoint}")
            connect_future = self.connection.connect()
            connect_future.result()
            
            self.is_connected = True
            logger.info("Successfully connected to AWS IoT Core")
            
            # Subscribe to default topics
            await self._subscribe_to_default_topics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to AWS IoT Core: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from AWS IoT Core."""
        if self.connection and self.is_connected:
            logger.info("Disconnecting from AWS IoT Core")
            disconnect_future = self.connection.disconnect()
            disconnect_future.result()
            self.is_connected = False
            logger.info("Disconnected from AWS IoT Core")
    
    def _on_connection_interrupted(self, connection, error, **kwargs):
        """Handle connection interruption."""
        logger.warning(f"AWS IoT connection interrupted: {error}")
        self.is_connected = False
    
    def _on_connection_resumed(self, connection, return_code, session_present, **kwargs):
        """Handle connection resumption."""
        logger.info("AWS IoT connection resumed")
        self.is_connected = True
    
    async def _subscribe_to_default_topics(self):
        """Subscribe to default topics."""
        default_topics = [
            self.topics.get('search_requests', 'arxplorer/search/requests'),
            self.topics.get('embeddings', 'arxplorer/embeddings')
        ]
        
        for topic in default_topics:
            await self.subscribe(topic, self._default_message_handler)
    
    async def subscribe(self, topic: str, handler: Callable[[str, Dict[str, Any]], None]):
        """Subscribe to a topic with a message handler."""
        if not self.is_connected:
            logger.error("Cannot subscribe: not connected to AWS IoT")
            return False
            
        try:
            self.message_handlers[topic] = handler
            
            def on_message_received(topic_received, payload, dup, qos, retain, **kwargs):
                """Handle incoming messages."""
                try:
                    message = json.loads(payload.decode('utf-8'))
                    self.executor.submit(handler, topic_received, message)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON message on {topic_received}: {e}")
                except Exception as e:
                    logger.error(f"Error handling message on {topic_received}: {e}")
            
            logger.info(f"Subscribing to topic: {topic}")
            subscribe_future, packet_id = self.connection.subscribe(
                topic=topic,
                qos=mqtt.QoS.AT_LEAST_ONCE,
                callback=on_message_received
            )
            subscribe_future.result()
            
            logger.info(f"Successfully subscribed to {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {topic}: {e}")
            return False
    
    async def publish(self, topic: str, message: Dict[str, Any], qos: Optional[int] = None) -> bool:
        """Publish a message to a topic."""
        if not self.is_connected:
            logger.error("Cannot publish: not connected to AWS IoT")
            return False
            
        try:
            qos_level = qos or self.qos
            payload = json.dumps(message, default=str)
            
            logger.debug(f"Publishing to {topic}: {payload}")
            
            publish_future, packet_id = self.connection.publish(
                topic=topic,
                payload=payload,
                qos=mqtt.QoS.AT_LEAST_ONCE if qos_level == 1 else mqtt.QoS.AT_MOST_ONCE
            )
            publish_future.result()
            
            logger.debug(f"Successfully published to {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            return False
    
    async def publish_search_request(self, query: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        """Publish a search request."""
        topic = self.topics.get('search_requests', 'arxplorer/search/requests')
        message = {
            'timestamp': asyncio.get_event_loop().time(),
            'query': query,
            'filters': filters or {},
            'client_id': self.thing_name
        }
        return await self.publish(topic, message)
    
    async def publish_search_results(self, query: str, results: list, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Publish search results."""
        topic = self.topics.get('search_results', 'arxplorer/search/results')
        message = {
            'timestamp': asyncio.get_event_loop().time(),
            'query': query,
            'results': results,
            'count': len(results),
            'metadata': metadata or {},
            'client_id': self.thing_name
        }
        return await self.publish(topic, message)
    
    async def publish_status(self, status: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Publish status update."""
        topic = self.topics.get('status', 'arxplorer/status')
        message = {
            'timestamp': asyncio.get_event_loop().time(),
            'status': status,
            'details': details or {},
            'client_id': self.thing_name
        }
        return await self.publish(topic, message)
    
    def _default_message_handler(self, topic: str, message: Dict[str, Any]):
        """Default message handler for incoming messages."""
        logger.info(f"Received message on {topic}: {message}")
        
        # Handle search requests
        if topic == self.topics.get('search_requests', 'arxplorer/search/requests'):
            query = message.get('query', '')
            if query:
                logger.info(f"Received search request: {query}")
                # TODO: Integrate with search engine
                # This would call your existing search functionality
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class MQTTSearchBridge:
    """Bridge between MQTT client and search functionality."""
    
    def __init__(self, mqtt_client: AWSIoTMQTTClient, search_engine=None):
        """Initialize the MQTT search bridge."""
        self.mqtt_client = mqtt_client
        self.search_engine = search_engine
        
    async def setup(self):
        """Setup the bridge and subscribe to relevant topics."""
        if not self.mqtt_client.enabled:
            return
            
        # Subscribe to search requests
        search_topic = self.mqtt_client.topics.get('search_requests', 'arxplorer/search/requests')
        await self.mqtt_client.subscribe(search_topic, self.handle_search_request)
        
    async def handle_search_request(self, topic: str, message: Dict[str, Any]):
        """Handle incoming search requests via MQTT."""
        try:
            query = message.get('query', '')
            filters = message.get('filters', {})
            client_id = message.get('client_id', 'unknown')
            
            logger.info(f"Processing search request from {client_id}: {query}")
            
            if self.search_engine:
                # Perform search using existing search engine
                results = await self.search_engine.search(query, **filters)
                
                # Publish results back
                await self.mqtt_client.publish_search_results(
                    query=query,
                    results=results,
                    metadata={'processed_by': self.mqtt_client.thing_name}
                )
                
                logger.info(f"Published {len(results)} search results for query: {query}")
            else:
                logger.warning("No search engine configured for MQTT bridge")
                
        except Exception as e:
            logger.error(f"Error handling search request: {e}")
            await self.mqtt_client.publish_status(
                status='error',
                details={'error': str(e), 'query': message.get('query', '')}
            )