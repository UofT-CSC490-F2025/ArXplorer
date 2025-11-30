"""
ArXplorer Search Module
Hybrid search combining SciBERT + FAISS with LLM judge + AWS IoT MQTT integration
"""

from .hybrid_search import HybridSearchEngine, SearchMode, SearchResult, SearchMetrics
from .a4_judge_integration import EnsembleJudge, JudgeResult, OllamaJudge, GRPOJudge
from .unified_api import SearchAPI

# MQTT integration (optional)
try:
    from ..aws import AWSIoTMQTTClient, MQTTSearchBridge
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

__all__ = [
    'HybridSearchEngine',
    'SearchMode', 
    'SearchResult',
    'SearchMetrics',
    'EnsembleJudge',
    'JudgeResult',
    'OllamaJudge',
    'GRPOJudge', 
    'SearchAPI'
]

if MQTT_AVAILABLE:
    __all__.extend(['AWSIoTMQTTClient', 'MQTTSearchBridge'])