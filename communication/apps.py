from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class CommunicationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'communication'
    verbose_name = 'Communication Module'
    
    def ready(self):
        """Initialize communication module when Django starts"""
        try:
            # Import signal handlers
            from . import signals
            
            # Initialize protocol registry
            from .core.registry import ProtocolRegistry
            registry = ProtocolRegistry()
            registry.initialize()
            
            logger.info("Communication module initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize communication module: {e}")
            raise
