"""
ArXplorer Data Processing Pipeline
Implements data ingestion, cleaning, transformation, and indexing for arXiv papers
"""

import asyncio
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import xml.etree.ElementTree as ET

# External dependencies
import requests
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Local imports
from src.core.schemas import (
    ArXivPaper, ProcessedPaper, PaperEmbedding, Author, 
    PaperStatus, PipelineConfig, DATABASE_SCHEMAS
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class ArXivDataIngester:
    """Handles data ingestion from arXiv API"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
    def fetch_papers(self, 
                    start_date: datetime,
                    end_date: datetime,
                    categories: Optional[List[str]] = None,
                    max_results: int = 1000) -> Iterator[ArXivPaper]:
        """
        Fetch papers from arXiv API within date range and categories
        
        Args:
            start_date: Start date for paper submission
            end_date: End date for paper submission  
            categories: List of arXiv categories to fetch
            max_results: Maximum number of papers to fetch
            
        Yields:
            ArXivPaper: Parsed paper objects
        """
        categories = categories or self.config.arxiv_categories
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Format dates for arXiv API
        start_str = start_date.strftime("%Y%m%d%H%M%S")
        end_str = end_date.strftime("%Y%m%d%H%M%S")
        date_query = f"submittedDate:[{start_str} TO {end_str}]"
        
        query = f"({category_query}) AND {date_query}"
        
        # Paginate through results
        start_index = 0
        batch_size = min(self.config.batch_size, max_results)
        
        while start_index < max_results:
            params = {
                'search_query': query,
                'start': start_index,
                'max_results': batch_size,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                response = self.session.get(self.config.arxiv_api_url, params=params)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                entries = root.findall('{http://www.w3.org/2005/Atom}entry')
                
                if not entries:
                    break
                    
                for entry in entries:
                    try:
                        paper = self._parse_arxiv_entry(entry)
                        if paper:
                            yield paper
                    except Exception as e:
                        self.logger.error(f"Error parsing entry: {e}")
                        continue
                
                start_index += len(entries)
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error fetching papers: {e}")
                break
    
    def _parse_arxiv_entry(self, entry: ET.Element) -> Optional[ArXivPaper]:
        """Parse individual arXiv entry from XML"""
        try:
            # Extract basic information
            arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            
            # Extract authors
            authors = []
            for author_elem in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name = author_elem.find('{http://www.w3.org/2005/Atom}name').text
                authors.append(Author(name=name))
            
            # Extract categories
            categories = []
            for category_elem in entry.findall('{http://www.w3.org/2005/Atom}category'):
                categories.append(category_elem.get('term'))
            
            # Extract dates
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            submitted_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
            
            updated_elem = entry.find('{http://www.w3.org/2005/Atom}updated')
            updated_date = None
            if updated_elem is not None:
                updated_date = datetime.fromisoformat(updated_elem.text.replace('Z', '+00:00'))
            
            # Extract optional fields
            doi = None
            journal_ref = None
            comments = None
            
            for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                if link.get('title') == 'doi':
                    doi = link.get('href')
            
            comment_elem = entry.find('{http://arxiv.org/schemas/atom}comment', 
                                    namespaces={'arxiv': 'http://arxiv.org/schemas/atom'})
            if comment_elem is not None:
                comments = comment_elem.text
                
            journal_elem = entry.find('{http://arxiv.org/schemas/atom}journal_ref',
                                    namespaces={'arxiv': 'http://arxiv.org/schemas/atom'})
            if journal_elem is not None:
                journal_ref = journal_elem.text
            
            return ArXivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                submitted_date=submitted_date,
                updated_date=updated_date,
                doi=doi,
                journal_ref=journal_ref,
                comments=comments
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing entry: {e}")
            return None


class TextProcessor:
    """Handles text cleaning and feature extraction"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.logger = logging.getLogger(__name__)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep scientific notation
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\[\]\/\\\$\%\@\#\&\*\+\=\<\>\?]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def extract_keywords(self, title: str, abstract: str, top_k: int = 10) -> List[str]:
        """Extract key terms using TF-IDF"""
        combined_text = f"{title} {abstract}"
        cleaned_text = self.clean_text(combined_text)
        
        try:
            # Create a fresh TF-IDF vectorizer for each document to avoid feature mismatch
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=1)
            tfidf_matrix = tfidf.fit_transform([cleaned_text])
            feature_names = tfidf.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = np.argsort(tfidf_scores)[-top_k:][::-1]
            keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
            return keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
    
    def calculate_readability(self, text: str) -> float:
        """Calculate simple readability score (words per sentence)"""
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            if len(sentences) == 0:
                return 0.0
                
            return len(words) / len(sentences)
            
        except Exception:
            return 0.0
    
    def process_paper(self, paper: ArXivPaper) -> ProcessedPaper:
        """Process a single paper"""
        try:
            cleaned_title = self.clean_text(paper.title)
            cleaned_abstract = self.clean_text(paper.abstract)
            
            keywords = self.extract_keywords(cleaned_title, cleaned_abstract)
            
            word_count = len(word_tokenize(f"{cleaned_title} {cleaned_abstract}"))
            readability_score = self.calculate_readability(cleaned_abstract)
            
            return ProcessedPaper(
                arxiv_id=paper.arxiv_id,
                cleaned_title=cleaned_title,
                cleaned_abstract=cleaned_abstract,
                extracted_keywords=keywords,
                word_count=word_count,
                readability_score=readability_score
            )
            
        except Exception as e:
            self.logger.error(f"Error processing paper {paper.arxiv_id}: {e}")
            raise


class EmbeddingGenerator:
    """Generates semantic embeddings using transformer models"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model)
            self.model = AutoModel.from_pretrained(self.config.embedding_model)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Loaded model: {self.config.embedding_model}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"Error encoding text: {e}")
            raise
    
    def generate_embeddings(self, processed_paper: ProcessedPaper) -> PaperEmbedding:
        """Generate embeddings for a processed paper"""
        try:
            title_embedding = self._encode_text(processed_paper.cleaned_title)
            abstract_embedding = self._encode_text(processed_paper.cleaned_abstract)
            
            # Create combined embedding
            combined_text = f"{processed_paper.cleaned_title} {processed_paper.cleaned_abstract}"
            combined_embedding = self._encode_text(combined_text)
            
            return PaperEmbedding(
                arxiv_id=processed_paper.arxiv_id,
                title_embedding=title_embedding.tolist(),
                abstract_embedding=abstract_embedding.tolist(),
                combined_embedding=combined_embedding.tolist(),
                model_name=self.config.embedding_model,
                model_version="1.0",
                embedding_dimension=len(combined_embedding),
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings for {processed_paper.arxiv_id}: {e}")
            raise


class VectorIndexer:
    """Manages FAISS vector index for similarity search"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.index = None
        self.paper_ids = []
        self.logger = logging.getLogger(__name__)
    
    def build_index(self, embeddings: List[PaperEmbedding]) -> None:
        """Build FAISS index from embeddings"""
        try:
            if not embeddings:
                raise ValueError("No embeddings provided")
            
            # Extract embedding vectors and IDs
            vectors = np.array([emb.combined_embedding for emb in embeddings], dtype=np.float32)
            self.paper_ids = [emb.arxiv_id for emb in embeddings]
            
            dimension = vectors.shape[1]
            
            # Create FAISS index
            # For small datasets, use Flat index; for large datasets, use IVF
            if self.config.faiss_index_type == "IVF" and len(embeddings) >= self.config.n_clusters:
                # IVF (Inverted File) index for large datasets
                quantizer = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.config.n_clusters)
                
                # Train the index
                self.index.train(vectors)
                self.logger.info(f"Using IVF index with {self.config.n_clusters} clusters")
                
            else:
                # Flat index for smaller datasets
                self.index = faiss.IndexFlatIP(dimension)
                self.logger.info(f"Using Flat index for {len(embeddings)} embeddings")
            
            # Add vectors to index
            self.index.add(vectors)
            
            self.logger.info(f"Built FAISS index with {len(embeddings)} papers")
            
        except Exception as e:
            self.logger.error(f"Error building index: {e}")
            raise
    
    def save_index(self, filepath: str) -> None:
        """Save FAISS index to disk"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            faiss.write_index(self.index, filepath)
            
            # Save paper IDs mapping
            ids_path = filepath.replace('.index', '_ids.json')
            with open(ids_path, 'w') as f:
                json.dump(self.paper_ids, f)
                
            self.logger.info(f"Saved index to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self, filepath: str) -> None:
        """Load FAISS index from disk"""
        try:
            self.index = faiss.read_index(filepath)
            
            # Load paper IDs mapping
            ids_path = filepath.replace('.index', '_ids.json')
            with open(ids_path, 'r') as f:
                self.paper_ids = json.load(f)
                
            self.logger.info(f"Loaded index from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            raise


class ArXplorerPipeline:
    """Main data processing pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        if os.path.exists(config_path):
            # TODO: Load from YAML file
            self.config = PipelineConfig()
        else:
            self.config = PipelineConfig()
        
        # Initialize components
        self.ingester = ArXivDataIngester(self.config)
        self.processor = TextProcessor(self.config)
        self.embedder = EmbeddingGenerator(self.config)
        self.indexer = VectorIndexer(self.config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create data directories
        for path in [self.config.raw_data_path, self.config.processed_data_path, 
                    self.config.embeddings_path, self.config.index_path]:
            os.makedirs(path, exist_ok=True)
    
    def run_ingestion(self, 
                     start_date: datetime,
                     end_date: datetime,
                     max_papers: int = 1000) -> List[ArXivPaper]:
        """Run data ingestion pipeline"""
        self.logger.info(f"Starting ingestion from {start_date} to {end_date}")
        
        papers = []
        for paper in self.ingester.fetch_papers(start_date, end_date, max_results=max_papers):
            papers.append(paper)
            
            # Save raw data periodically
            if len(papers) % self.config.batch_size == 0:
                self._save_raw_papers(papers[-self.config.batch_size:])
        
        # Save remaining papers
        if papers:
            remaining = len(papers) % self.config.batch_size
            if remaining > 0:
                self._save_raw_papers(papers[-remaining:])
        
        self.logger.info(f"Ingested {len(papers)} papers")
        return papers
    
    def run_processing(self, papers: List[ArXivPaper]) -> List[ProcessedPaper]:
        """Run text processing pipeline"""
        self.logger.info(f"Processing {len(papers)} papers")
        
        processed_papers = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit processing jobs
            future_to_paper = {
                executor.submit(self.processor.process_paper, paper): paper
                for paper in papers
            }
            
            # Collect results
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    processed_paper = future.result()
                    processed_papers.append(processed_paper)
                    
                    # Update paper status
                    paper.status = PaperStatus.PROCESSED
                    paper.processed_at = datetime.now()
                    
                except Exception as e:
                    self.logger.error(f"Processing failed for {paper.arxiv_id}: {e}")
                    paper.status = PaperStatus.FAILED
                    paper.error_message = str(e)
        
        # Save processed data
        self._save_processed_papers(processed_papers)
        
        self.logger.info(f"Processed {len(processed_papers)} papers successfully")
        return processed_papers
    
    def run_embedding(self, processed_papers: List[ProcessedPaper]) -> List[PaperEmbedding]:
        """Run embedding generation pipeline"""
        self.logger.info(f"Generating embeddings for {len(processed_papers)} papers")
        
        embeddings = []
        
        # Process in batches to manage memory
        for i in range(0, len(processed_papers), self.config.batch_size):
            batch = processed_papers[i:i + self.config.batch_size]
            
            for paper in batch:
                try:
                    embedding = self.embedder.generate_embeddings(paper)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    self.logger.error(f"Embedding failed for {paper.arxiv_id}: {e}")
                    continue
            
            # Save batch
            self._save_embeddings(embeddings[-len(batch):])
            
            self.logger.info(f"Generated embeddings for batch {i//self.config.batch_size + 1}")
        
        self.logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def run_indexing(self, embeddings: List[PaperEmbedding]) -> None:
        """Run vector indexing pipeline"""
        self.logger.info(f"Building index for {len(embeddings)} embeddings")
        
        # Build FAISS index
        self.indexer.build_index(embeddings)
        
        # Save index to disk
        index_path = os.path.join(self.config.index_path, "papers.index")
        self.indexer.save_index(index_path)
        
        self.logger.info("Index building completed")
    
    def run_full_pipeline(self, 
                         start_date: datetime,
                         end_date: datetime,
                         max_papers: int = 1000) -> None:
        """Run the complete data processing pipeline"""
        self.logger.info("Starting full ArXplorer pipeline")
        
        try:
            # Step 1: Data Ingestion
            papers = self.run_ingestion(start_date, end_date, max_papers)
            
            # Step 2: Text Processing  
            processed_papers = self.run_processing(papers)
            
            # Step 3: Embedding Generation
            embeddings = self.run_embedding(processed_papers)
            
            # Step 4: Vector Indexing
            self.run_indexing(embeddings)
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _save_raw_papers(self, papers: List[ArXivPaper]) -> None:
        """Save raw papers to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_papers_{timestamp}.json"
        filepath = os.path.join(self.config.raw_data_path, filename)
        
        data = [paper.to_dict() for paper in papers]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_processed_papers(self, papers: List[ProcessedPaper]) -> None:
        """Save processed papers to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_papers_{timestamp}.json"
        filepath = os.path.join(self.config.processed_data_path, filename)
        
        data = [paper.to_dict() for paper in papers]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_embeddings(self, embeddings: List[PaperEmbedding]) -> None:
        """Save embeddings to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{timestamp}.json"
        filepath = os.path.join(self.config.embeddings_path, filename)
        
        data = [embedding.to_dict() for embedding in embeddings]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# CLI interface for running the pipeline
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ArXplorer Data Processing Pipeline")
    parser.add_argument("--start-date", type=str, required=True, 
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-papers", type=int, default=1000,
                       help="Maximum number of papers to process")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Run pipeline
    pipeline = ArXplorerPipeline(args.config)
    pipeline.run_full_pipeline(start_date, end_date, args.max_papers)
