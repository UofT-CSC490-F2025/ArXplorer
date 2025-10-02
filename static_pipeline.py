"""
Simplified ArXplorer Pipeline for Static Dataset Processing
Focus on processing the Kaggle arXiv dataset without scheduling
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from kaggle_loader import KaggleArXivLoader
from pipeline import TextProcessor, EmbeddingGenerator, VectorIndexer, PipelineConfig
from schemas import ArXivPaper, ProcessedPaper, PaperEmbedding

class StaticArXplorerPipeline:
    """Simplified pipeline for processing static arXiv dataset"""
    
    def __init__(self, dataset_path: str, output_dir: str = "./processed_data"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "index").mkdir(exist_ok=True)
        (self.output_dir / "stats").mkdir(exist_ok=True)
        
        # Initialize components
        self.config = PipelineConfig()
        self.loader = KaggleArXivLoader(dataset_path)
        self.processor = TextProcessor(self.config)
        self.embedder = EmbeddingGenerator(self.config)
        self.indexer = VectorIndexer(self.config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_dataset(self, sample_size: int = 10000) -> Dict[str, Any]:
        """Analyze the dataset to understand its structure and content"""
        self.logger.info(f"Analyzing dataset with sample size: {sample_size}")
        
        stats = {
            'total_papers_sampled': 0,
            'categories': {},
            'years': {},
            'authors_count': {},
            'sample_papers': [],
            'processing_errors': 0
        }
        
        for paper in self.loader.load_papers(limit=sample_size):
            stats['total_papers_sampled'] += 1
            
            # Count categories
            for category in paper.categories:
                main_cat = category.split('.')[0]
                stats['categories'][main_cat] = stats['categories'].get(main_cat, 0) + 1
            
            # Count years
            year = paper.submitted_date.year if paper.submitted_date else 2000
            stats['years'][year] = stats['years'].get(year, 0) + 1
            
            # Count authors
            author_count = len(paper.authors)
            stats['authors_count'][author_count] = stats['authors_count'].get(author_count, 0) + 1
            
            # Sample papers
            if len(stats['sample_papers']) < 10:
                stats['sample_papers'].append({
                    'id': paper.arxiv_id,
                    'title': paper.title[:100] + '...' if len(paper.title) > 100 else paper.title,
                    'categories': paper.categories,
                    'year': year,
                    'authors': len(paper.authors)
                })
        
        # Save analysis results
        analysis_file = self.output_dir / "stats" / "dataset_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info(f"Dataset analysis saved to {analysis_file}")
        return stats
    
    def process_papers(self, 
                      categories: Optional[List[str]] = None,
                      max_papers: int = 1000,
                      start_year: Optional[int] = None,
                      end_year: Optional[int] = None) -> List[ProcessedPaper]:
        """Process papers from the dataset with filtering"""
        
        self.logger.info(f"Processing papers with filters:")
        self.logger.info(f"  Categories: {categories}")
        self.logger.info(f"  Max papers: {max_papers}")
        self.logger.info(f"  Year range: {start_year}-{end_year}")
        
        processed_papers = []
        raw_papers = []
        
        # Load and filter papers
        for paper in self.loader.load_papers(limit=max_papers, categories=categories):
            # Year filtering
            if start_year or end_year:
                paper_year = paper.submitted_date.year if paper.submitted_date else 2000
                if start_year and paper_year < start_year:
                    continue
                if end_year and paper_year > end_year:
                    continue
            
            raw_papers.append(paper)
            
            try:
                processed_paper = self.processor.process_paper(paper)
                processed_papers.append(processed_paper)
                
                if len(processed_papers) % 100 == 0:
                    self.logger.info(f"Processed {len(processed_papers)} papers...")
                    
            except Exception as e:
                self.logger.error(f"Failed to process paper {paper.arxiv_id}: {e}")
                continue
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save processed papers
        processed_file = self.output_dir / "processed" / f"processed_papers_{timestamp}.json"
        with open(processed_file, 'w') as f:
            json.dump([p.to_dict() for p in processed_papers], f, indent=2)
        
        # Save raw papers for reference
        raw_file = self.output_dir / "processed" / f"raw_papers_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump([p.to_dict() for p in raw_papers], f, indent=2)
        
        self.logger.info(f"Saved {len(processed_papers)} processed papers to {processed_file}")
        return processed_papers
    
    def generate_embeddings(self, processed_papers: List[ProcessedPaper]) -> List[PaperEmbedding]:
        """Generate embeddings for processed papers"""
        self.logger.info(f"Generating embeddings for {len(processed_papers)} papers")
        
        embeddings = []
        batch_size = 10  # Smaller batches for memory management
        
        for i in range(0, len(processed_papers), batch_size):
            batch = processed_papers[i:i + batch_size]
            batch_embeddings = []
            
            for paper in batch:
                try:
                    embedding = self.embedder.generate_embeddings(paper)
                    batch_embeddings.append(embedding)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate embedding for {paper.arxiv_id}: {e}")
                    continue
            
            # Save batch embeddings
            if batch_embeddings:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_file = self.output_dir / "embeddings" / f"embeddings_batch_{i//batch_size}_{timestamp}.json"
                with open(batch_file, 'w') as f:
                    json.dump([e.to_dict() for e in batch_embeddings], f, indent=2)
                
                self.logger.info(f"Saved batch {i//batch_size + 1} embeddings to {batch_file}")
        
        self.logger.info(f"Generated {len(embeddings)} embeddings total")
        return embeddings
    
    def build_search_index(self, embeddings: List[PaperEmbedding]) -> str:
        """Build FAISS search index"""
        self.logger.info(f"Building search index for {len(embeddings)} embeddings")
        
        self.indexer.build_index(embeddings)
        
        # Save index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_file = self.output_dir / "index" / f"papers_index_{timestamp}.index"
        self.indexer.save_index(str(index_file))
        
        self.logger.info(f"Search index saved to {index_file}")
        return str(index_file)
    
    def create_summary_report(self, 
                            processed_papers: List[ProcessedPaper],
                            embeddings: List[PaperEmbedding],
                            index_file: Optional[str] = None) -> Dict[str, Any]:
        """Create a summary report of the processing results"""
        
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'papers_processed': len(processed_papers),
            'embeddings_generated': len(embeddings),
            'index_created': index_file is not None,
            'index_file': index_file,
            'categories_found': {},
            'top_keywords': {},
            'processing_stats': {
                'avg_word_count': sum(p.word_count for p in processed_papers) / len(processed_papers) if processed_papers else 0,
                'avg_readability': sum(p.readability_score or 0 for p in processed_papers) / len(processed_papers) if processed_papers else 0,
                'papers_with_keywords': sum(1 for p in processed_papers if p.extracted_keywords)
            }
        }
        
        # Count categories
        for paper in processed_papers:
            # We need to get categories from the original papers
            # For now, we'll skip this or modify to get from processed data
            pass
        
        # Count most common keywords
        all_keywords = []
        for paper in processed_papers:
            all_keywords.extend(paper.extracted_keywords)
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        summary['top_keywords'] = dict(sorted(keyword_counts.items(), 
                                            key=lambda x: x[1], reverse=True)[:20])
        
        # Save summary
        summary_file = self.output_dir / "stats" / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary report saved to {summary_file}")
        return summary


def main():
    """Main function to demonstrate the simplified pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Static ArXplorer Pipeline")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to Kaggle arXiv dataset")
    parser.add_argument("--categories", nargs='+', 
                       default=["cs.AI", "cs.CL", "cs.LG"],
                       help="Categories to process")
    parser.add_argument("--max-papers", type=int, default=500,
                       help="Maximum papers to process")
    parser.add_argument("--start-year", type=int, 
                       help="Start year filter")
    parser.add_argument("--end-year", type=int,
                       help="End year filter")
    parser.add_argument("--skip-embeddings", action="store_true",
                       help="Skip embedding generation")
    parser.add_argument("--skip-index", action="store_true",
                       help="Skip index building")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze dataset, don't process")
    parser.add_argument("--output-dir", type=str, default="./arxplorer_output",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = StaticArXplorerPipeline(args.dataset_path, args.output_dir)
    
    if args.analyze_only:
        # Just analyze the dataset
        stats = pipeline.analyze_dataset(sample_size=10000)
        
        print("\n=== Dataset Analysis ===")
        print(f"Papers sampled: {stats['total_papers_sampled']}")
        print(f"Top categories: {dict(list(sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True))[:10])}")
        print(f"Year range: {min(stats['years'].keys())} - {max(stats['years'].keys())}")
        print(f"Sample papers: {len(stats['sample_papers'])}")
        return
    
    try:
        # Step 1: Analyze dataset
        print("Analyzing dataset...")
        stats = pipeline.analyze_dataset()
        
        # Step 2: Process papers
        print("Processing papers...")
        processed_papers = pipeline.process_papers(
            categories=args.categories,
            max_papers=args.max_papers,
            start_year=args.start_year,
            end_year=args.end_year
        )
        
        if not processed_papers:
            print("No papers were processed successfully")
            return
        
        embeddings = []
        index_file = None
        
        # Step 3: Generate embeddings (optional)
        if not args.skip_embeddings:
            print("Generating embeddings...")
            embeddings = pipeline.generate_embeddings(processed_papers)
            
            # Step 4: Build search index (optional)
            if embeddings and not args.skip_index:
                print("Building search index...")
                index_file = pipeline.build_search_index(embeddings)
        
        # Step 5: Create summary report
        print("Creating summary report...")
        summary = pipeline.create_summary_report(processed_papers, embeddings, index_file)
        
        print("\n=== Processing Complete ===")
        print(f"Papers processed: {len(processed_papers)}")
        print(f"Embeddings generated: {len(embeddings)}")
        print(f"Search index: {'Created' if index_file else 'Skipped'}")
        print(f"Output directory: {args.output_dir}")
        print(f"Top keywords: {list(summary['top_keywords'].keys())[:5]}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()