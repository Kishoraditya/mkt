from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views, api_views

app_name = 'monitoring'

# Create a router for API viewsets
router = DefaultRouter()
router.register(r'events', views.MonitoringEventViewSet)
router.register(r'metrics', views.PerformanceMetricViewSet)

# API URLs
api_urlpatterns = [
    path('events/', api_views.MonitoringEventListCreateAPIView.as_view(), name='event_list_api'),
    path('metrics/', api_views.PerformanceMetricListCreateAPIView.as_view(), name='metric_list_api'),
    path('stats/', api_views.monitoring_stats_api_view, name='monitoring_stats_api'),
    path('log-event/', api_views.log_event_api_view, name='log_event_api'),
]

urlpatterns = [
    path('api/', include(api_urlpatterns)),
    path('dashboard/', views.monitoring_dashboard, name='dashboard'),
]

