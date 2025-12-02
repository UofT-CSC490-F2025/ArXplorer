"""
Utility functions for ArXplorer evaluation.
Handles API queries, arXiv ID normalization, and result parsing.
"""

import requests
import time
import json
import re
from typing import List, Dict, Optional, Any
from pathlib import Path


def clean_latex_text(text: str) -> str:
    """
    Clean LaTeX escape sequences from text.
    
    Common patterns in arXiv data:
    - \\v{c} -> c (caron)
    - \\'a -> a (acute accent)
    - \\`e -> e (grave accent)
    - \\~n -> n (tilde)
    - \\^o -> o (circumflex)
    - \\"u -> u (umlaut)
    - \\c{c} -> c (cedilla)
    - \\l -> l (stroke)
    """
    if not text:
        return text
    
    # Replace common LaTeX accent commands
    replacements = [
        # Braced accents: \v{c}, \c{c}, etc.
        (r"\\[vcu`'^~\"]=?\{([a-zA-Z])\}", r"\1"),
        # Non-braced accents: \'a, \`e, etc.
        (r"\\[vcu`'^~\"]([a-zA-Z])", r"\1"),
        # Special characters
        (r"\\l\b", "l"),  # \l -> l
        (r"\\o\b", "o"),  # \o -> o
        (r"\\ae\b", "ae"),
        (r"\\AE\b", "AE"),
        (r"\\ss\b", "ss"),
        # Math mode cleanup (simple cases)
        (r"\$([^$]+)\$", r"\1"),
        # Remaining backslashes for common cases
        (r"\\&", "&"),
        (r"\\%", "%"),
        (r"\\_", "_"),
        # Remove remaining single backslashes before letters (be conservative)
        (r"\\([a-zA-Z])", r"\1"),
    ]
    
    cleaned = text
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
    
    # Clean up any remaining escaped braces
    cleaned = cleaned.replace("\\{", "{").replace("\\}", "}")
    
    return cleaned


def query_api(
    query: str,
    api_endpoint: str,
    top_k: int = 10,
    rerank: bool = False,
    rewrite: bool = True,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Query the ArXplorer API and return results.
    
    Args:
        query: Search query string
        api_endpoint: API endpoint URL (e.g., "http://3.96.123.45:8080")
        top_k: Number of results to return
        rerank: Enable reranking (disabled by default on cloud)
        rewrite: Enable LLM query rewriting
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with 'results' list and metadata
    """
    url = f"{api_endpoint}/api/v1/query"
    
    payload = {
        "query": query,
        "top_k": top_k,
        "rerank": rerank,
        "rewrite_query": rewrite
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"  ⚠ Query timed out: {query[:50]}...")
        return {"results": [], "error": "timeout"}
    except requests.exceptions.RequestException as e:
        print(f"  ✗ API error: {e}")
        return {"results": [], "error": str(e)}


def extract_arxiv_id(doc_id: str) -> str:
    """
    Normalize arXiv ID to format without prefix.
    
    Examples:
        "arxiv:1706.03762" -> "1706.03762"
        "1706.03762" -> "1706.03762"
        "https://arxiv.org/abs/1706.03762" -> "1706.03762"
    """
    if doc_id.startswith("arxiv:"):
        return doc_id[6:]  # Remove "arxiv:" prefix
    elif doc_id.startswith("http"):
        return doc_id.split("/")[-1]
    return doc_id


def load_arxiv_metadata(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load arXiv metadata from JSONL file.
    
    Returns:
        Dictionary mapping arXiv ID -> paper metadata
    """
    print(f"Loading arXiv metadata from {jsonl_path}...")
    metadata = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                paper = json.loads(line)
                arxiv_id = extract_arxiv_id(paper.get('id', ''))
                
                if arxiv_id:
                    # Clean LaTeX from title and authors
                    title = clean_latex_text(paper.get('title', ''))
                    authors = paper.get('authors', [])
                    if isinstance(authors, list):
                        authors = [clean_latex_text(author) for author in authors]
                    
                    # Extract year from published_date
                    # Format: "Sun, 1 Apr 2007 13:06:50 GMT"
                    year = None
                    published_date = paper.get('published_date', '')
                    if published_date:
                        # Try to extract 4-digit year from date string
                        year_match = re.search(r'\b(19|20)\d{2}\b', published_date)
                        if year_match:
                            year = year_match.group(0)
                    
                    metadata[arxiv_id] = {
                        'title': title,
                        'abstract': paper.get('abstract', ''),
                        'authors': authors,
                        'categories': paper.get('categories', []),
                        'year': year
                    }
            except json.JSONDecodeError:
                print(f"  ⚠ Skipping invalid JSON at line {i+1}")
                continue
    
    print(f"✓ Loaded metadata for {len(metadata)} papers")
    return metadata


def check_paper_in_results(target_id: str, results: List[Dict]) -> tuple[bool, Optional[int]]:
    """
    Check if target paper is in results and return (found, rank).
    
    Args:
        target_id: Target arXiv ID (without prefix)
        results: List of result dictionaries with 'doc_id' field
        
    Returns:
        (found: bool, rank: int or None) - rank is 1-indexed
    """
    target_id = extract_arxiv_id(target_id)
    
    for i, result in enumerate(results):
        result_id = extract_arxiv_id(result.get('doc_id', ''))
        if result_id == target_id:
            return True, i + 1  # 1-indexed rank
    
    return False, None


def calculate_reciprocal_rank(target_id: str, results: List[Dict]) -> float:
    """
    Calculate reciprocal rank for a single query.
    
    Returns:
        1/rank if found, 0 if not found
    """
    found, rank = check_paper_in_results(target_id, results)
    if found:
        return 1.0 / rank
    return 0.0


def get_api_endpoint_from_terraform() -> Optional[str]:
    """Get API endpoint from terraform output."""
    import subprocess
    
    terraform_dir = Path(__file__).parent.parent.parent / "terraform"
    try:
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


def test_api_connection(api_endpoint: str) -> bool:
    """Test if API is reachable."""
    try:
        response = requests.get(f"{api_endpoint}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✓ API is reachable at {api_endpoint}")
            if health.get('milvus_connected'):
                print("✓ API connected to Milvus")
            else:
                print("⚠ API not connected to Milvus")
            return True
        else:
            print(f"⚠ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Could not connect to API: {e}")
        return False
