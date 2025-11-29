"""
Fetch citation counts from Semantic Scholar API for arXiv papers.

Usage:
    python scripts/fetch_citations.py --data-file data/arxiv_1k.jsonl --output data/citations.json

Features:
- Public API rate limiting (100 requests per 5 minutes)
- Checkpointing (resume from interruptions)
- Progress tracking with ETA
- Handles missing papers gracefully
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import requests
from tqdm import tqdm


class SemanticScholarFetcher:
    """Fetch citation data from Semantic Scholar public API."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    RATE_LIMIT = 90  # Conservative: 90 requests per window (safety margin)
    RATE_WINDOW = 300  # 5 minutes in seconds
    MIN_REQUEST_INTERVAL = 0.6  # Minimum 0.6 seconds between requests
    
    def __init__(self, checkpoint_file: Optional[str] = None):
        self.checkpoint_file = checkpoint_file
        self.request_count = 0
        self.window_start = time.time()
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ArXivSearchResearchProject/1.0 (Educational Use)'
        })
    
    def _wait_if_needed(self):
        """Implement rate limiting with per-request delay."""
        # Ensure minimum time between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - time_since_last)
        
        self.request_count += 1
        self.last_request_time = time.time()
        
        # Check if we've hit the rate limit for this window
        if self.request_count >= self.RATE_LIMIT:
            elapsed = time.time() - self.window_start
            if elapsed < self.RATE_WINDOW:
                wait_time = self.RATE_WINDOW - elapsed + 10  # +10 sec buffer
                print(f"\n⏳ Rate limit reached ({self.request_count} requests). Waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
            
            # Reset counter and window
            self.request_count = 0
            self.window_start = time.time()
    
    def fetch_paper(self, arxiv_id: str) -> Optional[Dict]:
        """
        Fetch citation data for a single arXiv paper.
        
        Args:
            arxiv_id: arXiv ID (e.g., "0704.0047")
            
        Returns:
            Dict with citation data or None if not found
        """
        self._wait_if_needed()
        
        # Semantic Scholar accepts arXiv IDs in format: arXiv:ID
        paper_id = f"arXiv:{arxiv_id}"
        url = f"{self.BASE_URL}/{paper_id}"
        
        # Request specific fields to minimize response size
        params = {
            'fields': 'citationCount,year,venue,title,externalIds'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'citation_count': data.get('citationCount', 0),
                    'year': data.get('year'),
                    'venue': data.get('venue'),
                    'title': data.get('title'),
                    'fetched_at': datetime.now().isoformat()
                }
            elif response.status_code == 404:
                # Paper not found in Semantic Scholar
                return None
            elif response.status_code == 429:
                # Rate limit hit - back off significantly
                print(f"\n⚠️  Rate limit 429 for {arxiv_id}, waiting 60s...")
                time.sleep(60)
                # Reset window after 429
                self.request_count = 0
                self.window_start = time.time()
                return self.fetch_paper(arxiv_id)  # Retry
            else:
                print(f"\n⚠️  Error {response.status_code} for {arxiv_id}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"\n⚠️  Network error for {arxiv_id}: {e}")
            return None
    
    def load_checkpoint(self) -> Dict[str, Dict]:
        """Load existing citations from checkpoint file."""
        if self.checkpoint_file and Path(self.checkpoint_file).exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_checkpoint(self, citations: Dict[str, Dict]):
        """Save citations to checkpoint file."""
        if self.checkpoint_file:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(citations, f, indent=2)


def load_arxiv_ids(jsonl_file: str) -> list:
    """Load arXiv IDs from JSONL file."""
    ids = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc['id'])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Fetch citation counts from Semantic Scholar")
    parser.add_argument('--data-file', type=str, default='data/arxiv_1k.jsonl',
                        help='Input JSONL file with arXiv papers')
    parser.add_argument('--output', type=str, default='data/citations.json',
                        help='Output JSON file for citation data')
    parser.add_argument('--checkpoint', type=str, default='data/citations_checkpoint.json',
                        help='Checkpoint file (resume from interruptions)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-fetch all papers (ignore checkpoint)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SEMANTIC SCHOLAR CITATION FETCHER")
    print("="*60)
    print(f"Data file: {args.data_file}")
    print(f"Output: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    
    # Load arXiv IDs
    print("Loading arXiv IDs...")
    arxiv_ids = load_arxiv_ids(args.data_file)
    print(f"✓ Found {len(arxiv_ids)} papers")
    
    # Initialize fetcher
    fetcher = SemanticScholarFetcher(checkpoint_file=args.checkpoint)
    
    # Load checkpoint (unless --force)
    if args.force:
        citations = {}
        print("⚠️  Force mode: ignoring existing checkpoint")
    else:
        citations = fetcher.load_checkpoint()
        if citations:
            print(f"✓ Loaded checkpoint with {len(citations)} existing citations")
    
    # Filter out already-fetched IDs
    remaining_ids = [aid for aid in arxiv_ids if aid not in citations]
    print(f"Fetching citations for {len(remaining_ids)} papers...")
    print(f"Rate limit: {fetcher.RATE_LIMIT} requests per {fetcher.RATE_WINDOW//60} minutes")
    print(f"Request interval: {fetcher.MIN_REQUEST_INTERVAL}s between requests")
    print(f"Estimated time: ~{(len(remaining_ids) * fetcher.MIN_REQUEST_INTERVAL / 60):.1f} minutes")
    print()
    
    # Fetch citations with progress bar
    found_count = 0
    missing_count = 0
    error_count = 0
    
    try:
        with tqdm(total=len(remaining_ids), desc="Fetching") as pbar:
            for arxiv_id in remaining_ids:
                result = fetcher.fetch_paper(arxiv_id)
                
                if result is not None:
                    citations[arxiv_id] = result
                    found_count += 1
                    pbar.set_postfix({
                        'found': found_count,
                        'missing': missing_count,
                        'errors': error_count
                    })
                elif result is None:
                    # Track missing papers
                    citations[arxiv_id] = {
                        'citation_count': 0,
                        'year': None,
                        'venue': None,
                        'title': None,
                        'fetched_at': datetime.now().isoformat(),
                        'status': 'not_found'
                    }
                    missing_count += 1
                
                pbar.update(1)
                
                # Save checkpoint every 50 papers
                if (found_count + missing_count) % 50 == 0:
                    fetcher.save_checkpoint(citations)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving checkpoint...")
        fetcher.save_checkpoint(citations)
        print(f"✓ Saved {len(citations)} citations to {args.checkpoint}")
        print("Resume by running the same command again.")
        return
    
    # Save final results
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total papers: {len(arxiv_ids)}")
    print(f"Found in Semantic Scholar: {found_count}")
    print(f"Not found: {missing_count}")
    print(f"Already in checkpoint: {len(citations) - found_count - missing_count}")
    print()
    
    # Save to output file
    with open(args.output, 'w') as f:
        json.dump(citations, f, indent=2)
    print(f"✓ Saved citations to {args.output}")
    
    # Save metadata
    metadata = {
        'total_papers': len(arxiv_ids),
        'found_count': found_count,
        'missing_count': missing_count,
        'last_updated': datetime.now().isoformat(),
        'source': 'Semantic Scholar Public API',
        'data_file': args.data_file
    }
    
    metadata_file = args.output.replace('.json', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")
    
    # Print citation statistics
    cite_counts = [c['citation_count'] for c in citations.values() if c['citation_count'] > 0]
    if cite_counts:
        print()
        print("CITATION STATISTICS")
        print("-"*60)
        print(f"Papers with citations: {len(cite_counts)}")
        print(f"Mean citations: {sum(cite_counts) / len(cite_counts):.1f}")
        print(f"Median citations: {sorted(cite_counts)[len(cite_counts)//2]}")
        print(f"Max citations: {max(cite_counts)}")
        print(f"Min citations: {min(cite_counts)}")


if __name__ == "__main__":
    main()
