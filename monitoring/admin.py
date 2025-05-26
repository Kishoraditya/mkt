from django.contrib import admin
from .models import MonitoringEvent, PerformanceMetric


@admin.register(MonitoringEvent)
class MonitoringEventAdmin(admin.ModelAdmin):
    """Admin interface for monitoring events."""
    
    list_display = ('event_type', 'source', 'message', 'timestamp')
    list_filter = ('event_type', 'source', 'timestamp')
    search_fields = ('message', 'source')
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        (None, {
            'fields': ('event_type', 'source', 'message')
        }),
        ('Details', {
            'fields': ('details', 'timestamp')
        }),
    )


@admin.register(PerformanceMetric)
class PerformanceMetricAdmin(admin.ModelAdmin):
    """Admin interface for performance metrics."""
    
    list_display = ('name', 'value', 'unit', 'timestamp')
    list_filter = ('name', 'unit', 'timestamp')
    search_fields = ('name',)
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        (None, {
            'fields': ('name', 'value', 'unit')
        }),
        ('Details', {
            'fields': ('labels', 'timestamp')
        }),
    )
