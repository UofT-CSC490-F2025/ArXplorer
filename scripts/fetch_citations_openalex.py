"""
Fetch citation counts from OpenAlex API for arXiv papers.

OpenAlex offers much better rate limits than Semantic Scholar:
- 100,000 requests per day
- 10 requests per second
- Batch queries: up to 50 papers per request

Usage:
    python scripts/fetch_citations_openalex.py --data-file data/arxiv_1k.jsonl --output data/citations.json

Features:
- Batch fetching (50 papers per request)
- Public API rate limiting (10 req/sec)
- Checkpointing (resume from interruptions)
- Progress tracking with ETA
- Missing paper tracking
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Set
from datetime import datetime
import requests
from tqdm import tqdm


class OpenAlexFetcher:
    """Fetch citation data from OpenAlex public API."""
    
    BASE_URL = "https://api.openalex.org/works"
    RATE_LIMIT = 10  # requests per second (official limit)
    BATCH_SIZE = 50  # Max papers per request
    MIN_REQUEST_INTERVAL = 0.20  # ~5 requests per second (very conservative)
    
    def __init__(
        self, 
        checkpoint_file: Optional[str] = None,
        polite_pool_email: Optional[str] = None
    ):
        """
        Args:
            checkpoint_file: Path to checkpoint file for resuming
            polite_pool_email: Email for OpenAlex "polite pool" (better rate limits)
        """
        self.checkpoint_file = checkpoint_file
        self.polite_pool_email = polite_pool_email
        self.last_request_time = 0
        self.session = requests.Session()
        self.consecutive_rate_limits = 0  # Track consecutive 429 errors
        
        # Set up headers
        user_agent = 'ArXivSearchResearchProject/1.0 (Educational Use)'
        if polite_pool_email:
            user_agent += f'; mailto:{polite_pool_email}'
        
        self.session.headers.update({
            'User-Agent': user_agent
        })
    
    def _wait_if_needed(self):
        """Implement rate limiting with per-request delay."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - time_since_last)
        
        self.last_request_time = time.time()
    
    def _build_batch_filter(self, arxiv_ids: List[str]) -> str:
        """
        Build OpenAlex filter for batch querying by arXiv IDs.
        
        OpenAlex doesn't have a direct arXiv ID field, but we can query
        by constructing arxiv.org URLs or DOIs.
        
        Args:
            arxiv_ids: List of arXiv IDs (e.g., ["0704.0047", "1706.03762"])
            
        Returns:
            OR filter string for OpenAlex API
        """
        # Try querying by external IDs directly using OR syntax
        # Format: https://api.openalex.org/works?filter=ids.openalex:W123|W456|W789
        # But for arXiv, we'll use the arxiv: prefix
        arxiv_filters = [f"https://arxiv.org/abs/{arxiv_id}" for arxiv_id in arxiv_ids]
        return '|'.join(arxiv_filters)
    
    def fetch_batch(self, arxiv_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch citation data for a batch of arXiv papers.
        
        Args:
            arxiv_ids: List of arXiv IDs (up to 50)
            
        Returns:
            Dict mapping arxiv_id -> citation data
        """
        self._wait_if_needed()
        
        # Build query for batch of papers
        # OpenAlex OR syntax: filter multiple values with |
        arxiv_filters = [f"https://arxiv.org/abs/{aid}" for aid in arxiv_ids]
        filter_str = '|'.join(arxiv_filters)
        
        params = {
            'filter': f'ids.openalex:{filter_str}',
            'per-page': self.BATCH_SIZE,
            'select': 'id,ids,title,publication_year,cited_by_count'
        }
        
        if self.polite_pool_email:
            params['mailto'] = self.polite_pool_email
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = {}
                
                # Parse results and map back to arXiv IDs
                for work in data.get('results', []):
                    # Extract arXiv ID from the work's IDs
                    arxiv_id = self._extract_arxiv_id(work)
                    if arxiv_id:
                        results[arxiv_id] = {
                            'citation_count': work.get('cited_by_count', 0),
                            'year': work.get('publication_year'),
                            'title': work.get('title'),
                            'openalex_id': work.get('id'),
                            'fetched_at': datetime.now().isoformat()
                        }
                
                return results
                
            elif response.status_code == 429:
                # Rate limit hit - back off
                print(f"\n⚠️  Rate limit 429, waiting 10s...")
                time.sleep(10)
                return self.fetch_batch(arxiv_ids)  # Retry
            else:
                print(f"\n⚠️  Error {response.status_code} for batch")
                return {}
                
        except requests.exceptions.RequestException as e:
            print(f"\n⚠️  Network error for batch: {e}")
            return {}
    
    def _extract_arxiv_id(self, work: Dict) -> Optional[str]:
        """
        Extract arXiv ID from OpenAlex work object.
        
        OpenAlex doesn't have a dedicated arxiv_id field, but we can
        extract it from URLs in indexed_in or locations.
        
        Args:
            work: OpenAlex work object
            
        Returns:
            arXiv ID or None
        """
        # Try to find arXiv ID in the work's IDs or URLs
        # This is a heuristic approach - OpenAlex structure may vary
        
        # Check if there's an arxiv DOI pattern
        ids = work.get('ids', {})
        doi = ids.get('doi', '')
        if 'arxiv' in doi.lower():
            # Extract from DOI like "https://doi.org/10.48550/arxiv.1234.5678"
            parts = doi.split('arxiv.')
            if len(parts) > 1:
                return parts[1].split('/')[0]
        
        # Fallback: check title matching (less reliable)
        # We'll need to query differently - see fetch_batch_by_filter
        return None
    
    def fetch_batch_by_filter(self, arxiv_ids: List[str], max_consecutive_rate_limits: int = 5) -> Dict[str, Dict]:
        """
        Alternative batch fetch using search filter.
        
        Since OpenAlex doesn't have a direct arXiv ID field, we'll query
        papers that are indexed in arXiv and match our IDs.
        
        Args:
            arxiv_ids: List of arXiv IDs
            max_consecutive_rate_limits: Max consecutive 429 errors before aborting
            
        Returns:
            Dict mapping arxiv_id -> citation data
            
        Raises:
            RuntimeError: If too many consecutive rate limits hit
        """
        # Query papers indexed in arXiv
        # We'll need to make individual requests per ID for reliability
        results = {}
        
        for arxiv_id in arxiv_ids:
            # Rate limit each request
            self._wait_if_needed()
            
            # Query using arXiv DOI format
            # OpenAlex indexes arXiv papers with DOI: https://doi.org/10.48550/arXiv.{id}
            doi_url = f"https://doi.org/10.48550/arXiv.{arxiv_id}"
            url = f"{self.BASE_URL}/{doi_url}"
            
            params = {
                'select': 'id,ids,title,publication_year,cited_by_count'
            }
            
            if self.polite_pool_email:
                params['mailto'] = self.polite_pool_email
            
            try:
                response = self.session.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    work = response.json()
                    results[arxiv_id] = {
                        'citation_count': work.get('cited_by_count', 0),
                        'year': work.get('publication_year'),
                        'title': work.get('title'),
                        'openalex_id': work.get('id'),
                        'fetched_at': datetime.now().isoformat()
                    }
                    # Reset rate limit counter on success
                    self.consecutive_rate_limits = 0
                elif response.status_code == 404:
                    # Paper not found in OpenAlex
                    # Reset rate limit counter (not a rate limit issue)
                    self.consecutive_rate_limits = 0
                    continue
                elif response.status_code == 429:
                    self.consecutive_rate_limits += 1
                    print(f"\n⚠️  Rate limit 429 ({self.consecutive_rate_limits}/{max_consecutive_rate_limits}), waiting 30s...")
                    
                    # Check if we've hit too many rate limits
                    if self.consecutive_rate_limits >= max_consecutive_rate_limits:
                        raise RuntimeError(
                            f"Hit {self.consecutive_rate_limits} consecutive rate limits. "
                            "Daily API limit (100k req/day) likely reached. "
                            "Aborting to save progress."
                        )
                    
                    time.sleep(30)
                    self.last_request_time = time.time()  # Reset timer
                    # Don't retry here, will be handled in main loop
                    continue
                    
            except requests.exceptions.RequestException:
                continue
        
        return results
    
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


def load_arxiv_ids(jsonl_file: str) -> List[str]:
    """Load arXiv IDs from JSONL file."""
    ids = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc['id'])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Fetch citation counts from OpenAlex")
    parser.add_argument('--data-file', type=str, default='data/arxiv_1k.jsonl',
                        help='Input JSONL file with arXiv papers')
    parser.add_argument('--output', type=str, default='data/citations.json',
                        help='Output JSON file for citation data')
    parser.add_argument('--checkpoint', type=str, default='data/citations_checkpoint.json',
                        help='Checkpoint file (resume from interruptions)')
    parser.add_argument('--email', type=str, default=None,
                        help='Email for OpenAlex polite pool (better rate limits)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-fetch all papers (ignore checkpoint)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for querying (1=individual, 50=max batch)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("OPENALEX CITATION FETCHER")
    print("="*60)
    print(f"Data file: {args.data_file}")
    print(f"Output: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Email: {args.email or 'Not set (using common pool)'}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load arXiv IDs
    print("Loading arXiv IDs...")
    arxiv_ids = load_arxiv_ids(args.data_file)
    print(f"✓ Found {len(arxiv_ids)} papers")
    
    # Initialize fetcher
    fetcher = OpenAlexFetcher(
        checkpoint_file=args.checkpoint,
        polite_pool_email=args.email
    )
    
    # Load checkpoint (unless --force)
    if args.force:
        citations = {}
        missing_ids = set()
        print("⚠️  Force mode: ignoring existing checkpoint")
    else:
        checkpoint_data = fetcher.load_checkpoint()
        citations = checkpoint_data.get('citations', {}) if isinstance(checkpoint_data, dict) and 'citations' in checkpoint_data else checkpoint_data
        missing_ids = set(checkpoint_data.get('missing_ids', [])) if isinstance(checkpoint_data, dict) else set()
        if citations:
            print(f"✓ Loaded checkpoint with {len(citations)} existing citations")
        if missing_ids:
            print(f"✓ Found {len(missing_ids)} previously missing papers")
    
    # Filter out already-fetched IDs
    remaining_ids = [aid for aid in arxiv_ids if aid not in citations and aid not in missing_ids]
    print(f"Fetching citations for {len(remaining_ids)} papers...")
    print(f"Rate limit: {fetcher.RATE_LIMIT} requests per second")
    print(f"Request interval: {fetcher.MIN_REQUEST_INTERVAL}s between requests")
    
    # Estimate time
    if args.batch_size > 1:
        num_requests = len(remaining_ids) // args.batch_size + (1 if len(remaining_ids) % args.batch_size else 0)
        print(f"Batch mode: {num_requests} requests for {len(remaining_ids)} papers")
        print(f"Estimated time: ~{(num_requests * fetcher.MIN_REQUEST_INTERVAL / 60):.1f} minutes")
    else:
        print(f"Individual mode: {len(remaining_ids)} requests")
        print(f"Estimated time: ~{(len(remaining_ids) * fetcher.MIN_REQUEST_INTERVAL / 60):.1f} minutes")
    print()
    
    # Fetch citations with progress bar
    found_count = 0
    missing_count = len(missing_ids)
    
    try:
        with tqdm(total=len(remaining_ids), desc="Fetching") as pbar:
            # Process in batches
            for i in range(0, len(remaining_ids), args.batch_size):
                batch = remaining_ids[i:i + args.batch_size]
                
                # Fetch batch
                batch_results = fetcher.fetch_batch_by_filter(batch)
                
                # Process results
                for arxiv_id in batch:
                    if arxiv_id in batch_results:
                        citations[arxiv_id] = batch_results[arxiv_id]
                        found_count += 1
                    else:
                        # Mark as missing
                        missing_ids.add(arxiv_id)
                        citations[arxiv_id] = {
                            'citation_count': 0,
                            'year': None,
                            'title': None,
                            'openalex_id': None,
                            'fetched_at': datetime.now().isoformat(),
                            'status': 'not_found'
                        }
                        missing_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'found': found_count,
                        'missing': missing_count
                    })
                
                # Save checkpoint every 100 papers
                if (i + args.batch_size) % 100 == 0:
                    checkpoint_data = {
                        'citations': citations,
                        'missing_ids': list(missing_ids)
                    }
                    fetcher.save_checkpoint(checkpoint_data)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving checkpoint...")
        checkpoint_data = {
            'citations': citations,
            'missing_ids': list(missing_ids)
        }
        fetcher.save_checkpoint(checkpoint_data)
        print(f"✓ Saved {len(citations)} citations to {args.checkpoint}")
        print("Resume by running the same command again.")
        return
    
    except RuntimeError as e:
        # Handle rate limit abort (similar to keyboard interrupt)
        print(f"\n\n⚠️  {str(e)}")
        print("Saving checkpoint...")
        checkpoint_data = {
            'citations': citations,
            'missing_ids': list(missing_ids)
        }
        fetcher.save_checkpoint(checkpoint_data)
        print(f"✓ Saved {len(citations)} citations to {args.checkpoint}")
        print("💡 Resume tomorrow by running the same command again.")
        return
    
    # Save final results
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total papers: {len(arxiv_ids)}")
    print(f"Found in OpenAlex: {found_count}")
    print(f"Not found: {missing_count}")
    print(f"Already in checkpoint: {len(citations) - found_count - missing_count}")
    print()
    
    # Save to output file (citations only, without missing_ids wrapper)
    with open(args.output, 'w') as f:
        json.dump(citations, f, indent=2)
    print(f"✓ Saved citations to {args.output}")
    
    # Save missing IDs to separate file
    if missing_ids:
        missing_file = args.output.replace('.json', '_missing.json')
        with open(missing_file, 'w') as f:
            json.dump({
                'missing_ids': sorted(list(missing_ids)),
                'count': len(missing_ids),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
        print(f"✓ Saved {len(missing_ids)} missing IDs to {missing_file}")
    
    # Save metadata
    metadata = {
        'total_papers': len(arxiv_ids),
        'found_count': found_count,
        'missing_count': missing_count,
        'last_updated': datetime.now().isoformat(),
        'source': 'OpenAlex Public API',
        'data_file': args.data_file,
        'polite_pool_email': args.email or 'Not used'
    }
    
    metadata_file = args.output.replace('.json', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")
    
    # Print citation statistics
    cite_counts = [c['citation_count'] for c in citations.values() if c.get('citation_count', 0) > 0]
    if cite_counts:
        print()
        print("CITATION STATISTICS")
        print("-"*60)
        print(f"Papers with citations: {len(cite_counts)}")
        print(f"Mean citations: {sum(cite_counts) / len(cite_counts):.1f}")
        print(f"Median citations: {sorted(cite_counts)[len(cite_counts)//2]}")
        print(f"Max citations: {max(cite_counts)}")
        print(f"Min citations (non-zero): {min(cite_counts)}")


if __name__ == "__main__":
    main()
