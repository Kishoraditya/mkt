from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta
from .models import MonitoringEvent, PerformanceMetric
from .serializers import MonitoringEventSerializer, PerformanceMetricSerializer

import logging

logger = logging.getLogger(__name__)


class MonitoringEventListCreateAPIView(generics.ListCreateAPIView):
    """API view for listing and creating monitoring events."""
    
    serializer_class = MonitoringEventSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Get queryset with optional filtering."""
        queryset = MonitoringEvent.objects.all().order_by('-timestamp')
        
        # Filter by event type if provided
        event_type = self.request.query_params.get('event_type')
        if event_type:
            queryset = queryset.filter(event_type=event_type)
        
        # Filter by source if provided
        source = self.request.query_params.get('source')
        if source:
            queryset = queryset.filter(source=source)
        
        # Filter by date range
        hours = self.request.query_params.get('hours', 24)
        try:
            hours = int(hours)
            since = timezone.now() - timedelta(hours=hours)
            queryset = queryset.filter(timestamp__gte=since)
        except ValueError:
            pass
        
        return queryset[:100]  # Limit to 100 events


class PerformanceMetricListCreateAPIView(generics.ListCreateAPIView):
    """API view for listing and creating performance metrics."""
    
    serializer_class = PerformanceMetricSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Get queryset with optional filtering."""
        queryset = PerformanceMetric.objects.all().order_by('-timestamp')
        
        # Filter by metric name if provided
        name = self.request.query_params.get('name')
        if name:
            queryset = queryset.filter(name=name)
        
        # Filter by date range
        hours = self.request.query_params.get('hours', 24)
        try:
            hours = int(hours)
            since = timezone.now() - timedelta(hours=hours)
            queryset = queryset.filter(timestamp__gte=since)
        except ValueError:
            pass
        
        return queryset[:100]  # Limit to 100 metrics


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def monitoring_stats_api_view(request):
    """API view for monitoring statistics."""
    
    # Get event counts by type
    event_counts = MonitoringEvent.objects.values('event_type').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Get recent events
    recent_events = MonitoringEvent.objects.order_by('-timestamp')[:10]
    
    # Get performance metrics
    avg_response_time = PerformanceMetric.objects.filter(
        name='http_request',
        timestamp__gte=timezone.now() - timedelta(hours=1)
    ).aggregate(avg_value=Avg('value'))['avg_value'] or 0
    
    stats = {
        'total_events': MonitoringEvent.objects.count(),
        'event_counts': list(event_counts),
        'recent_events': MonitoringEventSerializer(recent_events, many=True).data,
        'avg_response_time': round(avg_response_time, 3),
        'system_status': 'healthy',
    }
    
    logger.info("Monitoring API: Retrieved monitoring statistics")
    return Response(stats)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def log_event_api_view(request):
    """API view for logging monitoring events."""
    
    serializer = MonitoringEventSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        logger.info(f"Monitoring API: Event logged - {serializer.data}")
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
