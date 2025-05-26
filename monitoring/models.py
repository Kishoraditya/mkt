from django.db import models
from django.utils import timezone

class MonitoringEvent(models.Model):
    """Model to store monitoring events."""
    
    EVENT_TYPES = (
        ('error', 'Error'),
        ('warning', 'Warning'),
        ('info', 'Info'),
        ('debug', 'Debug'),
        ('info', 'Information'),
        ('success', 'Success'),
        ('custom', 'Custom'),
    )
    
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    source = models.CharField(max_length=100)
    message = models.TextField()
    details = models.JSONField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.get_event_type_display()} - {self.source}: {self.message[:50]}"
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['event_type']),
            models.Index(fields=['source']),
            models.Index(fields=['timestamp']),
        ]


class PerformanceMetric(models.Model):
    """Model to store performance metrics."""
    
    name = models.CharField(max_length=100, help_text="Metric name")
    value = models.FloatField(help_text="Metric value")
    unit = models.CharField(max_length=20, default='', blank=True, help_text="Unit of measurement")
    labels = models.JSONField(default=dict, blank=True, help_text="Metric labels")
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.name}: {self.value} {self.unit}"
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['name', 'timestamp']),
        ]

class ActiveUser(models.Model):
    """Model for tracking active users."""
    
    session_key = models.CharField(max_length=40, unique=True)
    user_agent = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField()
    last_activity = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-last_activity']
    
    def __str__(self):
        return f"Session {self.session_key[:8]}... - {self.ip_address}"
    