"""
ArXplorer Cloud Frontend - Web interface for cloud-deployed API.

A lightweight Flask app that provides a modern UI while querying
the cloud-deployed Query API backend.
"""

import os
import sys
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import requests
from typing import Optional


app = Flask(__name__)
app.config['SECRET_KEY'] = 'arxplorer-cloud-frontend-key'

# Global API endpoint (will be auto-detected from terraform or set manually)
API_ENDPOINT = None


def get_api_endpoint_from_terraform() -> Optional[str]:
    """Get API endpoint from terraform output."""
    try:
        terraform_dir = Path(__file__).parent.parent / "terraform"
        result = subprocess.run(
            ["terraform", "output", "-raw", "query_api_endpoint"],
            cwd=terraform_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        endpoint = result.stdout.strip()
        if endpoint and not endpoint.startswith("Query API disabled"):
            return endpoint
    except Exception as e:
        print(f"Could not get API endpoint from terraform: {e}")
    return None


def get_api_endpoint() -> str:
    """Get the API endpoint, from environment variable, terraform, or default."""
    # Priority: Environment variable > Terraform > Manual config
    endpoint = os.environ.get('ARXPLORER_API_ENDPOINT')
    
    if not endpoint:
        endpoint = get_api_endpoint_from_terraform()
    
    if not endpoint:
        # Fallback to localhost (for testing)
        endpoint = "http://localhost:8080"
        print(f"Warning: Using default endpoint {endpoint}")
        print("Set ARXPLORER_API_ENDPOINT environment variable or run terraform to configure")
    
    return endpoint


@app.route('/')
def index():
    """Main search interface."""
    return render_template('index_cloud.html', api_endpoint=API_ENDPOINT)


@app.route('/search', methods=['POST'])
def search():
    """
    Proxy endpoint that forwards search requests to the cloud API.
    This allows CORS handling and adds a layer of validation.
    """
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'success': False, 'error': 'No query provided'}), 400
    
    query = data['query'].strip()
    if not query:
        return jsonify({'success': False, 'error': 'Empty query'}), 400
    
    # Parse search parameters
    top_k = min(int(data.get('top_k', 10)), 50)  # Limit to 50 results max
    rerank = data.get('enable_reranking', False)  # Default false since reranker disabled
    rewrite = data.get('enable_rewrite', True)
    
    try:
        # Forward to cloud API
        api_url = f"{API_ENDPOINT}/api/v1/query"
        
        response = requests.post(
            api_url,
            json={
                'query': query,
                'top_k': top_k,
                'rerank': rerank,
                'rewrite_query': rewrite
            },
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Transform API response to match frontend expectations
        formatted_result = {
            'success': True,
            'query': query,
            'total_results': len(result.get('results', [])),
            'results': [],
            'reranking_enabled': rerank,
            'intent': result.get('metadata', {}).get('intent'),
            'filter_expr': result.get('metadata', {}).get('filter_expr'),
            'query_variants': result.get('metadata', {}).get('rewritten_queries', [query]),
            'processing_time_ms': result.get('metadata', {}).get('processing_time_ms')
        }
        
        # Format results
        for i, paper in enumerate(result.get('results', [])):
            formatted_result['results'].append({
                'rank': i + 1,
                'id': paper.get('doc_id', ''),
                'title': paper.get('title', 'Untitled'),
                'abstract': paper.get('abstract', ''),
                'authors': paper.get('authors', []),
                'categories': paper.get('categories', []),
                'year': paper.get('year'),
                'citation_count': paper.get('citation_count', 0),
                'score': paper.get('score', 0.0),
                'dense_score': paper.get('dense_score'),
                'sparse_score': paper.get('sparse_score'),
                'cross_encoder_score': paper.get('cross_encoder_score')
            })
        
        return jsonify(formatted_result)
        
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'Search request timed out. The API may be processing a complex query.'
        }), 504
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'error': f'Could not connect to API at {API_ENDPOINT}. Make sure the Query API is running.'
        }), 503
        
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        return jsonify({
            'success': False,
            'error': f'API error: {error_detail}'
        }), e.response.status_code if e.response else 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """Health check endpoint that also checks API connectivity."""
    try:
        response = requests.get(f"{API_ENDPOINT}/health", timeout=5)
        api_healthy = response.status_code == 200
        api_status = response.json() if api_healthy else {}
    except Exception as e:
        api_healthy = False
        api_status = {'error': str(e)}
    
    return jsonify({
        'status': 'healthy',
        'frontend': 'running',
        'api_endpoint': API_ENDPOINT,
        'api_status': 'connected' if api_healthy else 'disconnected',
        'api_details': api_status
    })


@app.route('/config')
def get_config():
    """Return frontend configuration for debugging."""
    return jsonify({
        'api_endpoint': API_ENDPOINT,
        'environment': os.environ.get('FLASK_ENV', 'production')
    })


if __name__ == '__main__':
    # Initialize API endpoint
    API_ENDPOINT = get_api_endpoint()
    
    print("=" * 60)
    print("ArXplorer Cloud Frontend")
    print("=" * 60)
    print(f"API Endpoint: {API_ENDPOINT}")
    print("Frontend URL: http://localhost:5001")
    print("=" * 60)
    print()
    
    # Test API connectivity
    try:
        response = requests.get(f"{API_ENDPOINT}/health", timeout=5)
        if response.status_code == 200:
            print("✓ API is reachable and healthy")
            health_data = response.json()
            if health_data.get('milvus_connected'):
                print("✓ API connected to Milvus")
            else:
                print("⚠ API not connected to Milvus")
        else:
            print(f"⚠ API returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Could not connect to API: {e}")
        print("  Make sure the Query API is running and accessible")
    
    print()
    print("Starting frontend server...")
    
    # Run Flask app on different port than original frontend
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )
