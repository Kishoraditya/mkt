from django.apps import AppConfig


class MonitoringConfig(AppConfig):
    """Configuration for the monitoring app."""
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'monitoring'
    
    def ready(self):
        """Initialize app when Django starts."""
        # Import signals
        try:
            from . import signals
        except ImportError:
            pass
