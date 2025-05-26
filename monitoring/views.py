from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count, Max, Min
from django.utils import timezone
from datetime import timedelta

from rest_framework import viewsets, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from .models import MonitoringEvent, PerformanceMetric
from .serializers import MonitoringEventSerializer, PerformanceMetricSerializer

import logging

logger = logging.getLogger(__name__)


@login_required
def monitoring_dashboard(request):
    """View for the monitoring dashboard."""
    # Get recent events
    recent_events = MonitoringEvent.objects.order_by('-timestamp')[:20]
    
    # Get event counts by type
    event_counts = MonitoringEvent.objects.values('event_type').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Get performance metrics
    avg_response_time = PerformanceMetric.objects.filter(
        name='http_request',
        timestamp__gte=timezone.now() - timedelta(hours=1)
    ).aggregate(avg_value=Avg('value'))['avg_value'] or 0
    
    # Get total events count
    total_events = MonitoringEvent.objects.count()
    
    context = {
        'recent_events': recent_events,
        'event_counts': event_counts,
        'avg_response_time': round(avg_response_time, 3),
        'total_events': total_events,
    }
    
    return render(request, 'monitoring/dashboard.html', context)


class MonitoringEventViewSet(viewsets.ModelViewSet):
    """API viewset for monitoring events."""
    queryset = MonitoringEvent.objects.all()
    serializer_class = MonitoringEventSerializer
    permission_classes = [permissions.IsAuthenticated]
    filterset_fields = ['event_type', 'source']
    search_fields = ['message']
    ordering_fields = ['timestamp', 'event_type']
    ordering = ['-timestamp']


class PerformanceMetricViewSet(viewsets.ModelViewSet):
    """API viewset for performance metrics."""
    queryset = PerformanceMetric.objects.all()
    serializer_class = PerformanceMetricSerializer
    permission_classes = [permissions.IsAuthenticated]
    filterset_fields = ['name', 'unit']
    ordering_fields = ['timestamp', 'name', 'value']
    ordering = ['-timestamp']


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def monitoring_stats(request):
    """API view for monitoring statistics."""
    # Time range filter
    days = int(request.GET.get('days', 7))
    start_date = timezone.now() - timedelta(days=days)
    
    # Get event stats
    events = MonitoringEvent.objects.filter(timestamp__gte=start_date)
    event_stats = events.values('event_type').annotate(
        count=Count('id')
    ).order_by('event_type')
    
    # Get performance metrics stats
    metrics = PerformanceMetric.objects.filter(timestamp__gte=start_date)
    metric_stats = metrics.values('name').annotate(
        avg=Avg('value'),
        min=Min('value'),
        max=Max('value'),
        count=Count('id')
    ).order_by('name')
    
    # Prepare response
    stats = {
        'total_events': events.count(),
        'event_stats': list(event_stats),
        'total_metrics': metrics.count(),
        'metric_stats': list(metric_stats),
        'time_range': {
            'start': start_date,
            'end': timezone.now(),
            'days': days
        }
    }
    
    logger.info(f"Monitoring stats retrieved for {days} days")
    return Response(stats)


@csrf_exempt
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def log_event(request):
    """API endpoint to log a monitoring event."""
    serializer = MonitoringEventSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        logger.info(f"Monitoring event logged: {request.data.get('message')}")
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)
