"""
Custom Prometheus metrics for MKT application.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from django.conf import settings

# Blog metrics
BLOG_POSTS_CREATED = Counter(
    'mkt_blog_posts_total',
    'Total number of blog posts created',
    ['author', 'category']
)

BLOG_POST_VIEWS = Counter(
    'mkt_blog_views_total',
    'Total blog post views',
    ['post_slug', 'category']
)

BLOG_POST_READING_TIME = Histogram(
    'mkt_blog_reading_time_seconds',
    'Estimated reading time for blog posts',
    ['category'],
    buckets=[30, 60, 120, 300, 600, 900, 1800]  # 30s to 30min
)

# API metrics
API_REQUESTS = Counter(
    'mkt_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

API_RESPONSE_TIME = Histogram(
    'mkt_api_response_time_seconds',
    'API response time',
    ['endpoint', 'method']
)

# Monitoring metrics
MONITORING_EVENTS = Counter(
    'mkt_monitoring_events_total',
    'Total monitoring events',
    ['event_type', 'source']
)

ACTIVE_USERS = Gauge(
    'mkt_active_users',
    'Number of active users',
    ['time_period']
)

# System metrics
DATABASE_QUERY_TIME = Histogram(
    'mkt_database_query_duration_seconds',
    'Database query execution time',
    ['query_type', 'table']
)

CACHE_OPERATIONS = Counter(
    'mkt_cache_operations_total',
    'Cache operations',
    ['operation', 'result']
)

# Application info
APP_INFO = Info(
    'mkt_app_info',
    'Application information'
)

# Initialize app info
APP_INFO.info({
    'version': getattr(settings, 'APP_VERSION', '1.0.0'),
    'environment': 'development' if settings.DEBUG else 'production',
    'django_version': getattr(settings, 'DJANGO_VERSION', 'unknown')
})


def record_blog_post_created(author_name, category_name):
    """Record a new blog post creation."""
    BLOG_POSTS_CREATED.labels(
        author=author_name,
        category=category_name
    ).inc()


def record_blog_post_view(post_slug, category_name):
    """Record a blog post view."""
    BLOG_POST_VIEWS.labels(
        post_slug=post_slug,
        category=category_name
    ).inc()


def record_api_request(endpoint, method, status_code, response_time):
    """Record an API request."""
    API_REQUESTS.labels(
        endpoint=endpoint,
        method=method,
        status=status_code
    ).inc()
    
    API_RESPONSE_TIME.labels(
        endpoint=endpoint,
        method=method
    ).observe(response_time)


def record_monitoring_event(event_type, source):
    """Record a monitoring event."""
    MONITORING_EVENTS.labels(
        event_type=event_type,
        source=source
    ).inc()


def update_active_users(count, time_period='5m'):
    """Update active users count."""
    ACTIVE_USERS.labels(time_period=time_period).set(count)


def record_database_query(query_type, table_name, duration):
    """Record database query metrics."""
    DATABASE_QUERY_TIME.labels(
        query_type=query_type,
        table=table_name
    ).observe(duration)


def record_cache_operation(operation, result):
    """Record cache operation."""
    CACHE_OPERATIONS.labels(
        operation=operation,
        result=result
    ).inc()
