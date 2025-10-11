#!/usr/bin/env python3
"""
Data Source Manager - Switch between ArXiv API and Kaggle dataset
"""

import yaml
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from pipeline import ArXivDataIngester
from kaggle_loader import KaggleArXivLoader, KaggleArXplorerPipeline
from schemas import ArXivPaper, PipelineConfig

class DataSourceManager:
    """Manages different data sources for reproducible research"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str):
        """Load configuration"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get_data_source(self, source_type: str = "arxiv_api"):
        """
        Get appropriate data source
        
        Args:
            source_type: "arxiv_api" or "kaggle"
        """
        if source_type == "arxiv_api":
            return "live", ArXivDataIngester(PipelineConfig(**self.config.get('pipeline', {})))
        elif source_type == "kaggle":
            kaggle_path = self.config.get('data_sources', {}).get('kaggle_dataset_path')
            if not kaggle_path:
                raise ValueError("Kaggle dataset path not configured in config.yaml")
            return "static", KaggleArXivLoader(kaggle_path)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def fetch_papers_reproducible(self, 
                                max_results: int = 100,
                                source_type: str = "kaggle") -> List[ArXivPaper]:
        """
        Fetch papers with reproducible results
        
        Args:
            max_results: Maximum papers to fetch
            source_type: "arxiv_api" for live data, "kaggle" for static
        """
        source_nature, data_source = self.get_data_source(source_type)
        
        if source_nature == "static":
            # Kaggle dataset - reproducible
            papers = list(data_source.load_papers(max_papers=max_results))
            print(f"📊 Loaded {len(papers)} papers from Kaggle dataset (reproducible)")
            return papers
        else:
            # ArXiv API - live data
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Last week
            papers = list(data_source.fetch_papers(start_date, end_date, max_results=max_results))
            print(f"📊 Fetched {len(papers)} papers from ArXiv API (live data)")
            return papers

if __name__ == "__main__":
    # Example usage
    manager = DataSourceManager()
    
    print("🔄 Testing data sources...")
    
    # Test Kaggle (reproducible)
    try:
        papers_kaggle = manager.fetch_papers_reproducible(max_results=5, source_type="kaggle")
        print(f"✅ Kaggle: {len(papers_kaggle)} papers loaded")
    except Exception as e:
        print(f"❌ Kaggle source failed: {e}")
    
    # Test ArXiv API (live)
    try:
        papers_arxiv = manager.fetch_papers_reproducible(max_results=5, source_type="arxiv_api")
        print(f"✅ ArXiv API: {len(papers_arxiv)} papers loaded")
    except Exception as e:
        print(f"❌ ArXiv API failed: {e}")