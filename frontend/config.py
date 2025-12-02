"""
Frontend configuration for ArXplorer web interface.
"""

import os
from pathlib import Path


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'arxplorer-dev-secret-key'
    
    # Backend paths
    BACKEND_CONFIG_PATH = os.environ.get('BACKEND_CONFIG_PATH') or str(Path(__file__).parent.parent / "config.yaml")
    
    # Flask settings
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # Search settings
    MAX_RESULTS = int(os.environ.get('MAX_RESULTS', 50))
    DEFAULT_RESULTS = int(os.environ.get('DEFAULT_RESULTS', 10))
    SEARCH_TIMEOUT = int(os.environ.get('SEARCH_TIMEOUT', 30))  # seconds
    
    # Feature flags
    ENABLE_QUERY_REWRITING = os.environ.get('ENABLE_QUERY_REWRITING', 'True').lower() == 'true'
    ENABLE_RERANKING = os.environ.get('ENABLE_RERANKING', 'True').lower() == 'true'
    ENABLE_INTENT_BOOSTING = os.environ.get('ENABLE_INTENT_BOOSTING', 'True').lower() == 'true'


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'change-this-in-production'


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}