from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from blog.models import BlogPost
from .models import MonitoringEvent

import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender=BlogPost)
def log_blog_post_save(sender, instance, created, **kwargs):
    """Log when a blog post is created or updated."""
    action = "created" if created else "updated"
    
    # Log to monitoring events
    MonitoringEvent.objects.create(
        event_type='info',
        source='blog',
        message=f"Blog post '{instance.title}' was {action}",
        details={
            'post_id': instance.id,
            'post_slug': instance.slug,
            'author': str(instance.author) if instance.author else None,
            'action': action
        }
    )
    
    logger.info(f"Blog post '{instance.title}' was {action}")


@receiver(post_save, sender=User)
def log_user_save(sender, instance, created, **kwargs):
    """Log when a user is created or updated."""
    if created:
        # Log to monitoring events
        MonitoringEvent.objects.create(
            event_type='info',
            source='auth',
            message=f"User '{instance.username}' was created",
            details={
                'user_id': instance.id,
                'username': instance.username,
                'is_staff': instance.is_staff,
                'is_superuser': instance.is_superuser
            }
        )
        
        logger.info(f"User '{instance.username}' was created")
