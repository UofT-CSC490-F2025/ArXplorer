# AWS IoT MQTT Setup for ArXplorer

This document explains how to set up and use AWS IoT MQTT messaging with ArXplorer for distributed search requests and real-time communication.

## Overview

The AWS IoT MQTT integration allows ArXplorer to:
- Receive search requests via MQTT topics
- Publish search results to subscribing clients
- Enable real-time status updates and monitoring
- Support distributed search across multiple instances
- Integrate with IoT devices and mobile applications

## Prerequisites

1. **AWS Account** with IoT Core access
2. **AWS CLI** configured with appropriate credentials
3. **Python packages**: `boto3`, `awsiotsdk`, `paho-mqtt`

## Quick Setup

### 1. Install Dependencies

```bash
pip install boto3 awsiotsdk paho-mqtt
```

### 2. Run Setup Script

```bash
python setup_aws_iot.py --thing-name arxplorer-client --region us-east-1
```

This script will:
- Create an IoT Thing in AWS IoT Core
- Generate and download certificates
- Create IAM policies
- Configure AWS IoT endpoints
- Test the connection

### 3. Update Configuration

Add the generated configuration to your `config.yaml`:

```yaml
aws_iot:
  enabled: true
  region: "us-east-1"
  endpoint: "your-endpoint.iot.us-east-1.amazonaws.com"
  thing_name: "arxplorer-client"
  certificates:
    ca_cert: "certs/AmazonRootCA1.pem"
    cert_file: "certs/device-certificate.pem.crt"
    private_key: "certs/device-private.pem.key"
  topics:
    search_requests: "arxplorer/search/requests"
    search_results: "arxplorer/search/results"
    embeddings: "arxplorer/embeddings"
    status: "arxplorer/status"
  qos: 1
  keep_alive: 30
```

## Usage

### Basic MQTT Search Service

```python
from src.aws import MQTTSearchService

# Initialize and run the service
service = MQTTSearchService()
await service.run_forever()
```

### Manual MQTT Client

```python
from src.aws import AWSIoTMQTTClient

async def example():
    client = AWSIoTMQTTClient()
    await client.connect()
    
    # Publish search request
    await client.publish_search_request(
        query="machine learning transformers",
        filters={"category": "cs.LG"}
    )
    
    # Subscribe to results
    await client.subscribe("arxplorer/search/results", handle_results)
    
    await client.disconnect()
```

### Integration with Existing Search

```python
from src.aws import MQTTSearchBridge
from src.search import SearchAPI

# Create search engine
search_api = SearchAPI()
await search_api.initialize()

# Create MQTT client and bridge
mqtt_client = AWSIoTMQTTClient()
bridge = MQTTSearchBridge(mqtt_client, search_api)

await mqtt_client.connect()
await bridge.setup()
```

## MQTT Topics

### Search Requests (`arxplorer/search/requests`)
```json
{
  "timestamp": 1640995200.0,
  "query": "neural networks deep learning",
  "filters": {
    "category": "cs.LG",
    "max_results": 20,
    "mode": "hybrid"
  },
  "client_id": "mobile-app-123",
  "request_id": "req-456"
}
```

### Search Results (`arxplorer/search/results`)
```json
{
  "timestamp": 1640995201.5,
  "query": "neural networks deep learning",
  "results": [
    {
      "title": "Attention Is All You Need",
      "abstract": "The dominant sequence transduction models...",
      "authors": ["Ashish Vaswani", "Noam Shazeer"],
      "arxiv_id": "1706.03762",
      "score": 0.95
    }
  ],
  "count": 15,
  "metadata": {
    "processed_by": "arxplorer-client",
    "processing_time_ms": 234
  }
}
```

### Status Updates (`arxplorer/status`)
```json
{
  "timestamp": 1640995200.0,
  "status": "online",
  "details": {
    "service": "arxplorer_search",
    "capabilities": ["hybrid_search", "llm_judge"]
  },
  "client_id": "arxplorer-client"
}
```

## Security

### Certificate Management

- Certificates are stored in the `certs/` directory
- Private keys are automatically excluded from git
- Use AWS IoT certificate rotation for production
- Set appropriate file permissions (600 for private keys)

### Network Security

- All communication is encrypted using TLS 1.2+
- AWS IoT uses mutual TLS (mTLS) for authentication
- Certificates are tied to specific IoT policies
- Topic-level access control via IAM policies

### Best Practices

1. **Rotate certificates** regularly (every 90-365 days)
2. **Use least-privilege policies** for IoT things
3. **Monitor connection logs** in AWS CloudWatch
4. **Implement message validation** for incoming requests
5. **Use QoS 1 or 2** for important messages

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check AWS credentials and region
   - Verify certificate paths and permissions
   - Ensure IoT endpoint is correct

2. **Certificate Errors**
   - Verify certificate is active in AWS IoT
   - Check certificate and private key match
   - Ensure proper file permissions

3. **Message Not Received**
   - Check topic names and subscriptions
   - Verify IAM policies allow publish/subscribe
   - Check QoS settings

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('awscrt').setLevel(logging.DEBUG)
logging.getLogger('awsiot').setLevel(logging.DEBUG)
```

### AWS CloudWatch

Monitor your IoT communications:
- IoT Core Logs
- Connection metrics
- Message volume and errors

## Advanced Features

### Custom Message Handlers

```python
async def custom_handler(topic: str, message: dict):
    if topic.endswith('/embeddings'):
        # Handle embedding requests
        embeddings = await generate_embeddings(message['text'])
        await client.publish(f"{topic}/response", {"embeddings": embeddings})

await client.subscribe("arxplorer/custom/embeddings", custom_handler)
```

### Message Filtering

```python
# Server-side filtering using AWS IoT Rules
rule_sql = """
SELECT * FROM 'arxplorer/search/requests' 
WHERE query LIKE '%machine learning%'
"""
```

### Integration with Lambda

Connect MQTT messages to AWS Lambda for serverless processing:

```python
# Lambda function to process search requests
def lambda_handler(event, context):
    topic = event['topic']
    message = json.loads(event['message'])
    
    # Process search request
    results = perform_search(message['query'])
    
    # Publish results back
    iot_client.publish(
        topic='arxplorer/search/results',
        payload=json.dumps(results)
    )
```

## Production Deployment

### Scaling Considerations

1. **Connection Limits**: AWS IoT supports up to 1M concurrent connections per region
2. **Message Throughput**: Consider message size and frequency limits
3. **Certificate Management**: Automate certificate rotation
4. **Monitoring**: Set up CloudWatch alarms for connection health

### High Availability

1. **Multi-Region Setup**: Deploy across multiple AWS regions
2. **Connection Retry Logic**: Implement exponential backoff
3. **Health Checks**: Monitor service health via MQTT heartbeats
4. **Graceful Shutdown**: Handle disconnections properly

### Cost Optimization

1. **Message Batching**: Combine multiple requests when possible
2. **Compression**: Use message compression for large payloads
3. **Connection Pooling**: Reuse connections across requests
4. **Topic Design**: Optimize topic structure for efficient routing

## Examples

See `example_mqtt_usage.py` for complete working examples of:
- Basic MQTT client setup
- Search request/response handling
- Custom topic subscriptions
- Error handling and recovery