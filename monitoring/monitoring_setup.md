# Monitoring Setup Guide

This guide explains how to set up the monitoring system for the MKT application.

## Overview

The MKT monitoring system provides:

1. **Application Events**: User actions, system events, errors
2. **Performance Metrics**: Response times, database queries, cache hits
3. **System Health**: Server status, resource usage
4. **API Monitoring**: Endpoint performance and usage statistics

## Architecture

```bash
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Django App    │───▶│  Prometheus  │───▶│   Grafana   │
│                 │    │              │    │             │
│ - Events        │    │ - Metrics    │    │ - Dashboards│
│ - Metrics       │    │ - Time Series│    │ - Alerts    │
│ - Dashboard     │    │ - Storage    │    │ - Reports   │
└─────────────────┘    └──────────────┘    └─────────────┘
```

## Quick Setup

### Automated Setup (Recommended)

```bash
# Run the setup script
chmod +x setup_monitoring.sh
./setup_monitoring.sh

# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d
```

### Manual Setup

#### 1. Django Monitoring (Already Configured)

The monitoring app is integrated and provides:

- Event logging: `/monitoring/dashboard/`
- API endpoints: `/monitoring/api/`
- Prometheus metrics: `/metrics`

#### 2. Prometheus Setup

```bash
# Download Prometheus (if not done)
wget https://github.com/prometheus/prometheus/releases/download/v2.37.0/prometheus-2.37.0.linux-amd64.tar.gz
tar xvfz prometheus-2.37.0.linux-amd64.tar.gz

# Copy to project
cp prometheus-2.37.0.linux-amd64/prometheus monitoring/prometheus/
chmod +x monitoring/prometheus/prometheus

# Start Prometheus
./monitoring/prometheus/prometheus --config.file=prometheus.yml
```

#### 3. Grafana Setup (Docker)

```bash
docker run -d \
  --name=grafana \
  -p 3000:3000 \
  -v grafana-data:/var/lib/grafana \
  grafana/grafana:latest
```

## Configuration

### Prometheus Configuration

```yaml:prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mkt'
    static_configs:
      - targets: ['localhost:8001']
    scrape_interval: 5s
    metrics_path: '/metrics'
```

### Django Settings

Key monitoring settings in `mkt/settings/base.py`:

```python
# Prometheus metrics
PROMETHEUS_METRICS_EXPORT_PORT = 8001

# Monitoring app
INSTALLED_APPS = [
    # ... other apps
    'monitoring',
    'django_prometheus',
]

# Middleware for metrics collection
MIDDLEWARE = [
    'django_prometheus.middleware.PrometheusBeforeMiddleware',
    # ... other middleware
    'monitoring.middleware.PerformanceMiddleware',
    'django_prometheus.middleware.PrometheusAfterMiddleware',
]
```

## Available Metrics

### Django Application Metrics

- `django_http_requests_total` - Total HTTP requests
- `django_http_request_duration_seconds` - Request duration
- `django_db_query_duration_seconds` - Database query time
- `django_cache_operations_total` - Cache operations

### Custom Application Metrics

- `mkt_blog_posts_total` - Total blog posts
- `mkt_blog_views_total` - Total blog post views
- `mkt_monitoring_events_total` - Total monitoring events
- `mkt_api_requests_total` - API request counts

### System Metrics (via Node Exporter)

- CPU usage, memory usage, disk I/O
- Network statistics
- Process information

## Using the Monitoring System

### 1. Django Dashboard

Access: [http://localhost:8000/monitoring/dashboard/](http://localhost:8000/monitoring/dashboard/)

Features:

- Recent events timeline
- Event type distribution
- Performance metrics charts
- System health status

### 2. Prometheus

Access: [http://localhost:9090](http://localhost:9090)

Query examples:

```promql
# Request rate
rate(django_http_requests_total[5m])

# Average response time
rate(django_http_request_duration_seconds_sum[5m]) / rate(django_http_request_duration_seconds_count[5m])

# Error rate
rate(django_http_requests_total{status=~"5.."}[5m])
```

### 3. Grafana

Access: [http://localhost:3000](http://localhost:3000) (admin/admin)

Pre-configured dashboards:

- MKT Application Overview
- Blog Performance
- API Monitoring
- System Health

## Logging Custom Events

### Via Python Code

```python
from monitoring.models import MonitoringEvent

# Log an event
MonitoringEvent.objects.create(
    event_type='info',
    source='my_module',
    message='User performed action',
    details={'user_id': 123, 'action': 'login'}
)
```

### Via API

```bash
curl -X POST http://localhost:8000/monitoring/api/log-event/ \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "info",
    "source": "external_system",
    "message": "Data sync completed",
    "details": {"records": 1500}
  }'
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing in Prometheus**
   - Check Django server is running on port 8000
   - Verify `/metrics` endpoint is accessible
   - Check Prometheus configuration

2. **Grafana not showing data**
   - Verify Prometheus data source configuration
   - Check time range in dashboards
   - Ensure metrics are being collected

3. **High memory usage**
   - Adjust Prometheus retention settings
   - Implement metric filtering
   - Consider using recording rules

### Debug Commands

```bash
# Check Django metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana health
curl http://localhost:3000/api/health
```

## Performance Considerations

- Prometheus retention: Default 15 days
- Scrape interval: 15s (adjust based on needs)
- Event cleanup: Implement periodic cleanup for old events
- Metric cardinality: Avoid high-cardinality labels

## Security

- Restrict access to monitoring endpoints in production
- Use authentication for Grafana
- Secure Prometheus with reverse proxy
- Sanitize sensitive data in events
