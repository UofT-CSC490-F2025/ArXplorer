"""
ArXiv Kaggle Dataset Loader
Modified pipeline to work with the Kaggle arXiv dataset
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Iterator, Optional
import logging
from pathlib import Path

from schemas import ArXivPaper, Author, PaperStatus

class KaggleArXivLoader:
    """Load and parse the Kaggle arXiv dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        
    def load_papers(self, limit: Optional[int] = None, categories: Optional[List[str]] = None) -> Iterator[ArXivPaper]:
        """
        Load papers from the Kaggle arXiv dataset
        
        The Kaggle dataset is typically in JSON format with one paper per line
        Expected format:
        {
            "id": "0704.0001",
            "submitter": "Pavel Nadolsky",
            "authors": "C. Bal\\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan",
            "title": "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies",
            "comments": "37 pages, 15 figures; published version",
            "journal-ref": "Phys. Rev. D 76, 013009 (2007)",
            "doi": "10.1103/PhysRevD.76.013009",
            "abstract": "A fully differential calculation in perturbative QCD...",
            "categories": "hep-ph",
            "versions": [...],
            "update_date": "2007-05-23",
            "authors_parsed": [...]
        }
        """
        
        # Look for common dataset file names
        possible_files = [
            self.dataset_path / "arxiv-metadata-oai-snapshot.json",
            self.dataset_path / "arxiv.json", 
            self.dataset_path / "dataset.json"
        ]
        
        dataset_file = None
        for file_path in possible_files:
            if file_path.exists():
                dataset_file = file_path
                break
                
        if not dataset_file:
            # If no standard file found, look for any JSON file
            json_files = list(self.dataset_path.glob("*.json"))
            if json_files:
                dataset_file = json_files[0]
                self.logger.info(f"Using dataset file: {dataset_file}")
            else:
                raise FileNotFoundError(f"No JSON dataset file found in {self.dataset_path}")
        
        self.logger.info(f"Loading papers from {dataset_file}")
        
        count = 0
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if limit and count >= limit:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    paper = self._parse_kaggle_entry(data)
                    
                    if paper and self._filter_paper(paper, categories):
                        yield paper
                        count += 1
                        
                        if count % 1000 == 0:
                            self.logger.info(f"Loaded {count} papers...")
                            
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON on line {line_num + 1}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error parsing line {line_num + 1}: {e}")
                    continue
                    
        self.logger.info(f"Finished loading {count} papers")
    
    def _parse_kaggle_entry(self, data: Dict[str, Any]) -> Optional[ArXivPaper]:
        """Parse a single entry from the Kaggle dataset"""
        try:
            # Extract basic fields
            arxiv_id = data.get('id', '').replace('arXiv:', '')
            title = data.get('title', '').strip()
            abstract = data.get('abstract', '').strip()
            
            if not arxiv_id or not title or not abstract:
                return None
            
            # Parse authors
            authors = []
            authors_str = data.get('authors', '')
            if authors_str:
                # Split by common separators and clean up
                author_names = [name.strip() for name in authors_str.replace(' and ', ', ').split(',')]
                authors = [Author(name=name) for name in author_names if name]
            
            # Parse authors_parsed if available (more structured)
            if 'authors_parsed' in data and data['authors_parsed']:
                authors = []
                for author_data in data['authors_parsed']:
                    if isinstance(author_data, list) and len(author_data) >= 2:
                        # Format: [last_name, first_name, ...]
                        name = f"{author_data[1]} {author_data[0]}".strip()
                        authors.append(Author(name=name))
            
            # Parse categories
            categories = []
            categories_str = data.get('categories', '')
            if categories_str:
                categories = [cat.strip() for cat in categories_str.split()]
            
            # Parse dates
            submitted_date = None
            if 'versions' in data and data['versions']:
                # Use the first version's created date
                first_version = data['versions'][0]
                if 'created' in first_version:
                    try:
                        submitted_date = datetime.strptime(first_version['created'], '%a, %d %b %Y %H:%M:%S %Z')
                    except ValueError:
                        try:
                            # Try alternative format
                            submitted_date = datetime.fromisoformat(first_version['created'].replace('GMT', '+00:00'))
                        except ValueError:
                            pass
            
            # Fallback to update_date if available
            if not submitted_date and 'update_date' in data:
                try:
                    submitted_date = datetime.strptime(data['update_date'], '%Y-%m-%d')
                except ValueError:
                    submitted_date = datetime.now()  # Fallback
            
            if not submitted_date:
                submitted_date = datetime.now()
            
            # Extract optional fields
            doi = data.get('doi')
            journal_ref = data.get('journal-ref')
            comments = data.get('comments')
            
            return ArXivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                submitted_date=submitted_date,
                doi=doi,
                journal_ref=journal_ref,
                comments=comments,
                status=PaperStatus.RAW
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing entry: {e}")
            return None
    
    def _filter_paper(self, paper: ArXivPaper, categories: Optional[List[str]]) -> bool:
        """Filter papers by categories if specified"""
        if not categories:
            return True
            
        # Check if any of the paper's categories match the filter
        for paper_cat in paper.categories:
            for filter_cat in categories:
                if paper_cat.startswith(filter_cat):
                    return True
                    
        return False
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        stats = {
            'total_papers': 0,
            'categories': {},
            'years': {},
            'sample_papers': []
        }
        
        # Sample first 10000 papers for stats
        for i, paper in enumerate(self.load_papers(limit=10000)):
            stats['total_papers'] += 1
            
            # Count categories
            for cat in paper.categories:
                main_cat = cat.split('.')[0]  # Get main category (e.g., 'cs' from 'cs.AI')
                stats['categories'][main_cat] = stats['categories'].get(main_cat, 0) + 1
            
            # Count years
            year = paper.submitted_date.year
            stats['years'][year] = stats['years'].get(year, 0) + 1
            
            # Sample papers
            if len(stats['sample_papers']) < 5:
                stats['sample_papers'].append({
                    'id': paper.arxiv_id,
                    'title': paper.title[:100] + '...' if len(paper.title) > 100 else paper.title,
                    'categories': paper.categories,
                    'year': year
                })
        
        return stats


# Modified pipeline to work with Kaggle data
class KaggleArXplorerPipeline:
    """Modified pipeline for Kaggle arXiv dataset"""
    
    def __init__(self, dataset_path: str, config_path: str = "config.yaml"):
        from pipeline import ArXplorerPipeline
        
        # Initialize base pipeline
        self.base_pipeline = ArXplorerPipeline(config_path)
        self.loader = KaggleArXivLoader(dataset_path)
        self.logger = logging.getLogger(__name__)
    
    def run_kaggle_pipeline(self, 
                           categories: Optional[List[str]] = None,
                           max_papers: int = 1000,
                           skip_embeddings: bool = False) -> None:
        """Run pipeline on Kaggle dataset"""
        
        self.logger.info(f"Starting Kaggle ArXplorer pipeline")
        self.logger.info(f"Categories filter: {categories}")
        self.logger.info(f"Max papers: {max_papers}")
        
        try:
            # Step 1: Load papers from Kaggle dataset
            self.logger.info("Loading papers from Kaggle dataset...")
            papers = list(self.loader.load_papers(limit=max_papers, categories=categories))
            self.logger.info(f"Loaded {len(papers)} papers")
            
            if not papers:
                self.logger.warning("No papers loaded. Check dataset path and filters.")
                return
            
            # Step 2: Text Processing
            self.logger.info("Processing papers...")
            processed_papers = self.base_pipeline.run_processing(papers)
            
            if not skip_embeddings:
                # Step 3: Generate Embeddings (optional - requires large models)
                self.logger.info("Generating embeddings...")
                embeddings = self.base_pipeline.run_embedding(processed_papers)
                
                # Step 4: Build Search Index
                self.logger.info("Building search index...")
                self.base_pipeline.run_indexing(embeddings)
            else:
                self.logger.info("Skipping embeddings and indexing (skip_embeddings=True)")
            
            self.logger.info("Kaggle pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ArXplorer Pipeline for Kaggle Dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to Kaggle arXiv dataset directory")
    parser.add_argument("--categories", nargs='+', 
                       default=["cs.AI", "cs.CL", "cs.LG", "cs.IR"],
                       help="Categories to filter (e.g., cs.AI cs.CL)")
    parser.add_argument("--max-papers", type=int, default=1000,
                       help="Maximum number of papers to process")
    parser.add_argument("--skip-embeddings", action="store_true",
                       help="Skip embedding generation (faster for testing)")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only show dataset statistics")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.stats_only:
        # Just show dataset statistics
        loader = KaggleArXivLoader(args.dataset_path)
        stats = loader.get_dataset_stats()
        
        print("\n=== Dataset Statistics ===")
        print(f"Sample size analyzed: {stats['total_papers']} papers")
        print(f"\nTop categories:")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cat}: {count}")
        
        print(f"\nYear distribution:")
        for year, count in sorted(stats['years'].items())[-10:]:
            print(f"  {year}: {count}")
            
        print(f"\nSample papers:")
        for paper in stats['sample_papers']:
            print(f"  {paper['id']}: {paper['title']} ({paper['year']})")
    else:
        # Run the pipeline
        pipeline = KaggleArXplorerPipeline(args.dataset_path)
        pipeline.run_kaggle_pipeline(
            categories=args.categories,
            max_papers=args.max_papers,
            skip_embeddings=args.skip_embeddings
        )