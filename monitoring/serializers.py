from rest_framework import serializers
from .models import MonitoringEvent, PerformanceMetric


class MonitoringEventSerializer(serializers.ModelSerializer):
    """Serializer for monitoring events."""
    
    class Meta:
        model = MonitoringEvent
        fields = '__all__'
        read_only_fields = ['timestamp']


class PerformanceMetricSerializer(serializers.ModelSerializer):
    """Serializer for performance metrics."""
    
    class Meta:
        model = PerformanceMetric
        fields = '__all__'
        read_only_fields = ['timestamp']
