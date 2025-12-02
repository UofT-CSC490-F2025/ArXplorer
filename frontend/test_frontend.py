#!/usr/bin/env python3
"""
Test script for ArXplorer frontend.

Tests the search API endpoint and basic functionality.
"""

import requests
import json
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_frontend(base_url="http://localhost:5000"):
    """Test the frontend API endpoints."""
    
    print(f"Testing ArXplorer Frontend at {base_url}")
    print("="*50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Health check passed: {health_data}")
            if not health_data.get('search_system_ready'):
                print("⚠️  Warning: Search system not ready")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test home page
    print("\n2. Testing home page...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✓ Home page accessible")
        else:
            print(f"✗ Home page failed: {response.status_code}")
    except requests.RequestException as e:
        print(f"✗ Home page failed: {e}")
    
    # Test search endpoint
    print("\n3. Testing search endpoint...")
    test_queries = [
        "attention mechanism",
        "transformer architecture", 
        "neural networks"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n3.{i} Testing query: '{query}'")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/search",
                json={
                    "query": query,
                    "top_k": 5,
                    "enable_reranking": True
                },
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✓ Search successful ({end_time - start_time:.2f}s)")
                    print(f"  - Found {data.get('total_results', 0)} results")
                    print(f"  - Intent: {data.get('intent', 'unknown')}")
                    print(f"  - Query variants: {len(data.get('query_variants', []))}")
                    
                    # Show first result
                    results = data.get('results', [])
                    if results:
                        first = results[0]
                        print(f"  - Top result: {first.get('title', 'N/A')[:60]}...")
                        print(f"  - Score: {first.get('score', 0):.3f}")
                    else:
                        print("  - No results found")
                else:
                    print(f"✗ Search failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"✗ Search request failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error: {error_data}")
                except:
                    print(f"  Raw response: {response.text[:200]}")
                    
        except requests.RequestException as e:
            print(f"✗ Search request failed: {e}")
    
    print("\n" + "="*50)
    print("Frontend test completed!")
    return True


def test_search_features(base_url="http://localhost:5000"):
    """Test specific search features."""
    
    print(f"\nTesting Advanced Search Features")
    print("="*40)
    
    # Test with different parameters
    test_cases = [
        {
            "name": "High top_k",
            "params": {"query": "machine learning", "top_k": 20, "enable_reranking": True}
        },
        {
            "name": "No reranking",
            "params": {"query": "deep learning", "top_k": 10, "enable_reranking": False}
        },
        {
            "name": "Specific paper search",
            "params": {"query": "attention is all you need", "top_k": 5, "enable_reranking": True}
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/search",
                json=test_case['params'],
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✓ {test_case['name']} successful ({end_time - start_time:.2f}s)")
                    print(f"  - Results: {data.get('total_results', 0)}")
                    print(f"  - Reranking: {data.get('reranking_enabled', False)}")
                else:
                    print(f"✗ {test_case['name']} failed: {data.get('error')}")
            else:
                print(f"✗ {test_case['name']} request failed: {response.status_code}")
                
        except requests.RequestException as e:
            print(f"✗ {test_case['name']} request failed: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ArXplorer Frontend')
    parser.add_argument('--url', default='http://localhost:5000', help='Frontend URL')
    parser.add_argument('--advanced', action='store_true', help='Run advanced feature tests')
    
    args = parser.parse_args()
    
    # Run basic tests
    success = test_frontend(args.url)
    
    # Run advanced tests if requested
    if args.advanced and success:
        test_search_features(args.url)