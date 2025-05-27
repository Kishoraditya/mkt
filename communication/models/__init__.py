from django.db import models

# Create your models here.
"""
Django models for the communication module.
"""

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid
import json

class Agent(models.Model):
    """Model representing an AI agent in the communication system"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='agents')
    
    # Agent capabilities and metadata
    capabilities = models.JSONField(default=dict, help_text="Agent capabilities and features")
    metadata = models.JSONField(default=dict, help_text="Additional agent metadata")
    
    # Connection information
    endpoint_url = models.URLField(help_text="Agent's communication endpoint")
    supported_protocols = models.JSONField(default=list, help_text="List of supported protocols")
    
    # Status and lifecycle
    is_active = models.BooleanField(default=True)
    is_online = models.BooleanField(default=False)
    last_seen = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['owner', 'is_active']),
            models.Index(fields=['is_online', 'last_seen']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.id})"
    
    def update_last_seen(self):
        """Update the last seen timestamp"""
        self.last_seen = timezone.now()
        self.save(update_fields=['last_seen'])

class Conversation(models.Model):
    """Model representing a conversation between agents"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    participants = models.ManyToManyField(Agent, related_name='conversations')
    
    # Conversation metadata
    title = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    protocol_used = models.CharField(max_length=10, choices=[
        ('a2a', 'Agent-to-Agent'),
        ('acp', 'Agent Communication Protocol'),
        ('anp', 'Agent Network Protocol'),
    ])
    
    # Status
    is_active = models.BooleanField(default=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"Conversation {self.id} ({self.protocol_used})"

class CommunicationMessage(models.Model):
    """Model representing a message in the communication system"""
    
    MESSAGE_TYPES = [
        ('response', 'Response'),
        ('notification', 'Notification'),
        ('error', 'Error'),
        ('heartbeat', 'Heartbeat'),
    ]
    
    PROTOCOL_CHOICES = [
        ('a2a', 'Agent-to-Agent'),
        ('acp', 'Agent Communication Protocol'),
        ('anp', 'Agent Network Protocol'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('sent', 'Sent'),
        ('delivered', 'Delivered'),
        ('failed', 'Failed'),
        ('expired', 'Expired'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Message routing
    sender = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='sent_messages')
    receiver = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='received_messages')
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages', null=True, blank=True)
    
    # Message content
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES)
    protocol = models.CharField(max_length=10, choices=PROTOCOL_CHOICES)
    content = models.JSONField(help_text="Message content and payload")
    metadata = models.JSONField(default=dict, help_text="Message metadata and headers")
    
    # Message relationships
    reply_to = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='replies')
    
    # Status and delivery
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    retry_count = models.PositiveIntegerField(default=0)
    max_retries = models.PositiveIntegerField(default=3)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['sender', 'status']),
            models.Index(fields=['receiver', 'status']),
            models.Index(fields=['conversation', 'created_at']),
            models.Index(fields=['protocol', 'message_type']),
        ]
    
    def __str__(self):
        return f"Message {self.id} ({self.sender} -> {self.receiver})"
    
    def mark_as_sent(self):
        """Mark message as sent"""
        self.status = 'sent'
        self.sent_at = timezone.now()
        self.save(update_fields=['status', 'sent_at'])
    
    def mark_as_delivered(self):
        """Mark message as delivered"""
        self.status = 'delivered'
        self.delivered_at = timezone.now()
        self.save(update_fields=['status', 'delivered_at'])
    
    def mark_as_failed(self):
        """Mark message as failed"""
        self.status = 'failed'
        self.save(update_fields=['status'])
    
    def can_retry(self):
        """Check if message can be retried"""
        return self.retry_count < self.max_retries and self.status == 'failed'

class ProtocolConfiguration(models.Model):
    """Model for storing protocol-specific configurations"""
    
    protocol = models.CharField(max_length=10, choices=[
        ('a2a', 'Agent-to-Agent'),
        ('acp', 'Agent Communication Protocol'),
        ('anp', 'Agent Network Protocol'),
    ], unique=True)
    
    is_enabled = models.BooleanField(default=True)
    version = models.CharField(max_length=10, default='1.0')
    configuration = models.JSONField(default=dict, help_text="Protocol-specific configuration")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Protocol Configuration"
        verbose_name_plural = "Protocol Configurations"
    
    def __str__(self):
        return f"{self.get_protocol_display()} Configuration"

class AgentCapability(models.Model):
    """Model representing specific capabilities of agents"""
    
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    category = models.CharField(max_length=50, help_text="Capability category (e.g., 'nlp', 'vision', 'reasoning')")
    version = models.CharField(max_length=20, default='1.0')
    
    # Capability metadata
    input_schema = models.JSONField(default=dict, help_text="JSON schema for capability inputs")
    output_schema = models.JSONField(default=dict, help_text="JSON schema for capability outputs")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Agent Capabilities"
        ordering = ['category', 'name']
    
    def __str__(self):
        return f"{self.name} (v{self.version})"

class CommunicationEvent(models.Model):
    """Model for tracking communication events and analytics"""
    
    EVENT_TYPES = [
        ('agent_registered', 'Agent Registered'),
        ('agent_connected', 'Agent Connected'),
        ('agent_disconnected', 'Agent Disconnected'),
        ('message_sent', 'Message Sent'),
        ('message_received', 'Message Received'),
        ('message_failed', 'Message Failed'),
        ('protocol_error', 'Protocol Error'),
        ('authentication_failed', 'Authentication Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, null=True, blank=True)
    message = models.ForeignKey(CommunicationMessage, on_delete=models.CASCADE, null=True, blank=True)
    
    # Event details
    details = models.JSONField(default=dict, help_text="Event-specific details")
    error_message = models.TextField(blank=True)
    
    # Context
    protocol = models.CharField(max_length=10, choices=[
        ('a2a', 'Agent-to-Agent'),
        ('acp', 'Agent Communication Protocol'),
        ('anp', 'Agent Network Protocol'),
    ], null=True, blank=True)
    
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['event_type', 'timestamp']),
            models.Index(fields=['agent', 'timestamp']),
            models.Index(fields=['protocol', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.get_event_type_display()} - {self.timestamp}"
