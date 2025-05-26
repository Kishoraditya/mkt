"""
Enhanced middleware for collecting performance metrics.
"""
import time
import re
from prometheus_client import Counter, Histogram, Gauge
from django.utils.deprecation import MiddlewareMixin
import logging
from django.db import connection
from django.core.cache import cache
from .models import PerformanceMetric, MonitoringEvent
from .metrics import (
    record_api_request, 
    record_monitoring_event,
    record_database_query,
    update_active_users
)

logger = logging.getLogger(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    'django_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'django_http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'django_http_requests_active',
    'Active HTTP requests',
    ['method']
)

DB_QUERY_COUNT = Counter(
    'django_db_query_total',
    'Total database queries',
    ['view']
)

# Regex to normalize URLs with IDs
URL_PATTERN = re.compile(r'/(\d+)/')


class PrometheusMiddleware(MiddlewareMixin):
    """Middleware to collect Prometheus metrics."""
    
    def process_request(self, request):
        """Process request and collect metrics."""
        request._monitoring_start_time = time.time()
        request._monitoring_queries_before = len(connection.queries)
        request.start_time = time.time()
        method = request.method
        ACTIVE_REQUESTS.labels(method=method).inc()

    def process_response(self, request, response):
        """Process response and collect metrics."""
        if hasattr(request, 'start_time'):
            # Calculate request duration
            duration = time.time() - request.start_time
            
            # Normalize the URL path
            path = request.path
            normalized_path = URL_PATTERN.sub(r'/ID/', path)
            
            # Record metrics
            method = request.method
            status = response.status_code
            
            REQUEST_COUNT.labels(
                method=method,
                endpoint=normalized_path,
                status=status
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=normalized_path
            ).observe(duration)
            
            ACTIVE_REQUESTS.labels(method=method).dec()
            
            # Record DB query count if available
            if hasattr(request, '_num_queries'):
                view_name = request.resolver_match.view_name if request.resolver_match else 'unknown'
                DB_QUERY_COUNT.labels(view=view_name).inc(request._num_queries)

        """Record performance metrics."""
        if not hasattr(request, '_monitoring_start_time'):
            return response
        
        # Calculate metrics
        duration = time.time() - request._monitoring_start_time
        queries_count = len(connection.queries) - request._monitoring_queries_before
        
        # Record API metrics for API endpoints
        if request.path.startswith('/api/') or request.path.startswith('/blog/api/'):
            record_api_request(
                endpoint=request.path,
                method=request.method,
                status_code=response.status_code,
                response_time=duration
            )
        
        # Record performance metric in database (sample 10% of requests)
        if hash(request.path) % 10 == 0:  # Sample 10%
            try:
                PerformanceMetric.objects.create(
                    name='http_request',
                    value=duration,
                    unit='seconds',
                    labels={
                        'path': request.path,
                        'method': request.method,
                        'status_code': response.status_code,
                        'queries_count': queries_count
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record performance metric: {e}")
        
        # Log slow requests
        if duration > 2.0:  # Requests taking more than 2 seconds
            try:
                MonitoringEvent.objects.create(
                    event_type='warning',
                    source='performance_middleware',
                    message=f'Slow request detected: {request.path}',
                    details={
                        'duration': duration,
                        'queries_count': queries_count,
                        'method': request.method,
                        'status_code': response.status_code,
                        'user_agent': request.META.get('HTTP_USER_AGENT', '')[:200]
                    }
                )
                record_monitoring_event('warning', 'performance_middleware')
            except Exception as e:
                logger.error(f"Failed to log slow request: {e}")
        
        # Record database query metrics
        if queries_count > 0:
            try:
                avg_query_time = sum(
                    float(query['time']) for query in connection.queries[-queries_count:]
                ) / queries_count
                
                record_database_query(
                    query_type='select',  # Simplified for now
                    table_name='mixed',   # Simplified for now
                    duration=avg_query_time
                )
            except Exception as e:
                logger.error(f"Failed to record database metrics: {e}")
        
        return response
    
    def process_exception(self, request, exception):
        """Record exceptions as monitoring events."""
        try:
            MonitoringEvent.objects.create(
                event_type='error',
                source='performance_middleware',
                message=f'Exception in {request.path}: {str(exception)}',
                details={
                    'exception_type': exception.__class__.__name__,
                    'path': request.path,
                    'method': request.method,
                    'user_agent': request.META.get('HTTP_USER_AGENT', '')[:200]
                }
            )
            record_monitoring_event('error', 'performance_middleware')
        except Exception as e:
            logger.error(f"Failed to log exception: {e}")
        
        return None


import time
from django.utils import timezone
from django.utils.deprecation import MiddlewareMixin
from .models import MonitoringEvent, PerformanceMetric, ActiveUser


class PerformanceMiddleware(MiddlewareMixin):
    """Middleware to track request performance."""
    
    def process_request(self, request):
        request.start_time = time.time()
    
    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            
            # Record performance metric
            try:
                PerformanceMetric.objects.create(
                    name='http_request',
                    value=duration,
                    unit='seconds',
                    labels={
                        'method': request.method,
                        'path': request.path,
                        'status_code': response.status_code,
                    }
                )
            except Exception:
                pass  # Don't break the request if monitoring fails
        
        return response


class ActiveUsersMiddleware(MiddlewareMixin):
    """Middleware to track active users."""
    
    def process_request(self, request):
        if hasattr(request, 'session'):
            session_key = request.session.session_key
            if session_key:
                user_agent = request.META.get('HTTP_USER_AGENT', '')
                ip_address = self.get_client_ip(request)
                
                try:
                    ActiveUser.objects.update_or_create(
                        session_key=session_key,
                        defaults={
                            'user_agent': user_agent,
                            'ip_address': ip_address,
                            'last_activity': timezone.now(),
                        }
                    )
                except Exception:
                    pass  # Don't break the request if monitoring fails
    
    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR', '127.0.0.1')
        return ip
