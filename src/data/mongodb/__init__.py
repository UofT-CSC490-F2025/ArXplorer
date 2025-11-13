"""
MongoDB package for ArXplorer
Provides document-based storage and querying for research papers
"""

from .client import MongoDBClient
from .models import *
from .operations import *

__all__ = [
    'MongoDBClient',
    'PaperRepository',
    'EmbeddingRepository', 
    'SearchRepository',
    'PipelineRepository'
]