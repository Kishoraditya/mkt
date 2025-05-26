# Grafana Dashboard Setup

This document explains how to set up Grafana dashboards for monitoring the MKT application.

## Prerequisites

- Prometheus running and collecting metrics from the application
- Grafana installed and running

## Quick Start

### Using Docker (Recommended)

```bash
# Start Grafana with docker-compose
docker-compose -f docker-compose.monitoring.yml up -d grafana

# Or standalone
docker run -d \
  --name=mkt-grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana:latest
```

Access: [http://localhost:3000](http://localhost:3000) (admin/admin)

## Initial Configuration

### 1. Add Prometheus Data Source

1. Login to Grafana (admin/admin)
2. Go to Configuration → Data Sources
3. Click "Add data source"
4. Select "Prometheus"
5. Configure:
   - **URL**: `http://prometheus:9090` (Docker) or `http://localhost:9090` (local)
   - **Access**: Server (default)
6. Click "Save & Test"

### 2. Import MKT Dashboard

1. Go to Dashboards → Import
2. Upload `monitoring/dashboards/mkt_dashboard.json`
3. Select Prometheus data source
4. Click "Import"

## Available Dashboards

### 1. MKT Application Overview

**Panels:**

- Request Rate (requests/second)
- Response Time (average, 95th percentile)
- Error Rate (4xx, 5xx responses)
- Active Requests
- Database Query Time
- Cache Hit Rate

**Key Metrics:**

```promql
# Request rate
rate(django_http_requests_total[5m])

# Response time
histogram_quantile(0.95, rate(django_http_request_duration_seconds_bucket[5m]))

# Error rate
rate(django_http_requests_total{status=~"[45].."}[5m])
```

### 2. Blog Performance

**Panels:**

- Blog Post Views
- Popular Posts
- Category Distribution
- Author Activity
- Reading Time Analysis

**Key Metrics:**

```promql
# Blog views
increase(mkt_blog_views_total[1h])

# Post creation rate
rate(mkt_blog_posts_total[1d])
```

### 3. API Monitoring

**Panels:**

- API Endpoint Performance
- Request Volume by Endpoint
- API Error Rates
- Response Size Distribution

**Key Metrics:**

```promql
# API response time by endpoint
histogram_quantile(0.95, 
  rate(django_http_request_duration_seconds_bucket{path=~"/api/.*"}[5m])
) by (path)

# API request volume
sum(rate(django_http_requests_total{path=~"/api/.*"}[5m])) by (path)
```

### 4. System Health

**Panels:**

- Memory Usage
- CPU Usage
- Disk I/O
- Network Traffic
- Process Count

## Dashboard Variables

### Template Variables

Add these variables to make dashboards dynamic:

1. **Environment**
   - Type: Custom
   - Values: `development,staging,production`

2. **Time Range**
   - Type: Interval
   - Values: `5m,15m,1h,6h,24h`

3. **Instance**
   - Type: Query
   - Query: `label_values(django_http_requests_total, instance)`

## Custom Panels

### Creating a New Panel

1. Click "Add Panel" on dashboard
2. Select visualization type
3. Configure query:

    ```promql
    # Example: Blog post creation over time
    increase(mkt_blog_posts_total[1h])
    ```

4. Set panel options:
   - Title: "Blog Posts Created"
   - Unit: "Posts"
   - Legend: "{{author}}"

### Panel Types for MKT

**Stat Panels:**

- Total blog posts
- Total users
- System uptime

**Graph Panels:**

- Request rate over time
- Response time trends
- Error rate trends

**Table Panels:**

- Top blog posts by views
- Recent monitoring events
- API endpoint performance

**Heatmap Panels:**

- Response time distribution
- Request patterns by hour

## Alerting

### Setting Up Alerts

1. Edit a panel
2. Go to "Alert" tab
3. Configure conditions:

```yaml
# Example: High error rate alert
Condition: IS ABOVE 0.05
Evaluation: every 1m for 5m
```

### Notification Channels

**Email Notifications:**

1. Go to Alerting → Notification channels
2. Add new channel:
   - Type: Email
   - Addresses: [admin@example.com](admin@example.com)

**Slack Notifications:**

1. Create Slack webhook
2. Add notification channel:
   - Type: Slack
   - Webhook URL: your-slack-webhook

### Alert Rules Examples

```yaml
# High Error Rate
Alert Name: High Error Rate
Condition: avg() OF query(A, 5m, now) IS ABOVE 0.05
Message: Error rate is above 5% for 5 minutes

# Slow Response Time
Alert Name: Slow Response Time
Condition: avg() OF query(A, 5m, now) IS ABOVE 2
Message: Average response time is above 2 seconds

# Low Blog Activity
Alert Name: Low Blog Activity
Condition: sum() OF query(A, 1h, now) IS BELOW 1
Message: No blog posts created in the last hour
```

## Advanced Configuration

### Custom Themes

Create custom theme in `grafana.ini`:

```ini
[theme]
default_theme = custom

[custom_theme]
app_bg_color = #1e1e1e
panel_bg_color = #2d2d2d
```

### LDAP Integration

```ini
[auth.ldap]
enabled = true
config_file = /etc/grafana/ldap.toml
```

### Database Configuration

For persistent storage:

```yaml
# docker-compose.monitoring.yml
grafana:
  environment:
    - GF_DATABASE_TYPE=postgres
    - GF_DATABASE_HOST=db:5432
    - GF_DATABASE_NAME=grafana
    - GF_DATABASE_USER=grafana
    - GF_DATABASE_PASSWORD=password
```

## Backup and Restore

### Export Dashboard

```bash
# Export dashboard JSON
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:3000/api/dashboards/uid/mkt-dashboard
```

### Backup Grafana Data

```bash
# Backup Grafana database
docker exec mkt-grafana grafana-cli admin export-dashboard \
  --homeDashboard > backup.json
```

## Troubleshooting

### Common Issues

1. **No data in panels**
   - Check Prometheus data source connection
   - Verify metrics are being scraped
   - Check time range settings

2. **Slow dashboard loading**
   - Reduce query frequency
   - Optimize PromQL queries
   - Use recording rules for complex queries

3. **Alert not firing**
   - Check alert conditions
   - Verify notification channels
   - Check alert history

### Debug Queries

```bash
# Test Prometheus connection
curl http://localhost:9090/api/v1/query?query=up

# Check Grafana health
curl http://localhost:3000/api/health

# Test alert webhook
curl -X POST http://localhost:3000/api/alerts/test
```

## Best Practices

### Dashboard Design

1. **Keep it simple**: 6-8 panels per dashboard
2. **Use consistent colors**: Red for errors, green for success
3. **Add descriptions**: Help users understand metrics
4. **Group related panels**: Use rows to organize

### Query Optimization

1. **Use appropriate intervals**: Match scrape intervals
2. **Avoid high cardinality**: Limit label combinations
3. **Use recording rules**: For complex calculations
4. **Cache results**: Use query caching for repeated queries

### Maintenance

1. **Regular cleanup**: Remove unused dashboards
2. **Update queries**: Keep up with metric changes
3. **Monitor performance**: Check Grafana resource usage
4. **Backup regularly**: Export important dashboards

## Dashboard Panels

The MKT dashboard includes the following panels:

### Application Health

- Request Rate
- Error Rate
- Average Response Time
- Active Requests

### Database Performance

- Query Rate
- Average Query Time
- Slow Queries

### Blog Performance

- Blog Post Views
- API Request Rate
- Cache Hit/Miss Ratio

## Creating Custom Dashboards

To create a custom dashboard:

1. Click "Create" > "Dashboard"
2. Add panels using the "Add Panel" button
3. Configure each panel with Prometheus queries
4. Save the dashboard

## Example Queries

### Request Rate

```bash
rate(django_http_requests_total[5m])
```

### Error Rate

```bash
rate(django_http_requests_total{status=~"5.."}[5m])
```

### Average Response Time

```bash
rate(django_http_request_duration_seconds_sum[5m]) / rate(django_http_request_duration_seconds_count[5m])
```

### Blog Post Views

```bash
rate(blog_post_views_total[5m])
```

### 4. Create Tests for Monitoring App

```python:monitoring\tests.py
```
