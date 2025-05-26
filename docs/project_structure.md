# Project Structure

## Overview

The MKT project follows Django best practices with a modular application structure, clear separation of concerns, and maintainable code organization.

## Root Directory Structure

```bash
mkt-project/
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── manage.py                   # Django management script
├── setup.sh                    # Project setup script
├── docker-compose.yml          # Container orchestration
├── blog/                       # Blog application
├── home/                       # Home page application
├── monitoring/                 # Monitoring application
├── search/                     # Search functionality
├── mkt/                        # Project configuration
├── templates/                  # Global templates
├── static/                     # Global static files
├── media/                      # User uploaded files
├── logs/                       # Application logs
├── cache/                      # File-based cache
├── prometheus/                 # Prometheus configuration
├── grafana/                    # Grafana configuration
└── docs/                       # Project documentation
```

## Application Structure

### Blog Application (`blog/`)

```bash
blog/
├── __init__.py
├── admin.py                    # Admin configuration
├── apps.py                     # App configuration
├── models.py                   # Data models
├── views.py                    # Page views
├── api_views.py                # REST API views
├── serializers.py              # DRF serializers
├── urls.py                     # URL patterns
├── tests.py                    # Test cases
├── migrations/                 # Database migrations
│   ├── __init__.py
│   ├── 0001_initial.py
│   └── ...
├── templates/                  # Blog templates
│   └── blog/
│       ├── blog_index_page.html
│       ├── blog_post.html
│       └── components/
│           ├── post_card.html
│           └── pagination.html
├── static/                     # Blog static files
│   └── blog/
│       ├── css/
│       │   └── blog.css
│       ├── js/
│       │   └── blog.js
│       └── images/
└── management/                 # Custom management commands
    ├── __init__.py
    └── commands/
        ├── __init__.py
        └── create_sample_data.py
```

### Home Application (`home/`)

```bash
home/
├── __init__.py
├── admin.py
├── apps.py
├── models.py                   # HomePage model
├── views.py
├── urls.py
├── tests.py
├── migrations/
│   ├── __init__.py
│   ├── 0001_initial.py
│   └── ...
├── templates/
│   └── home/
│       ├── home_page.html
│       └── standard_page.html
└── static/
    └── home/
        ├── css/
        ├── js/
        └── images/
```

### Monitoring Application (`monitoring/`)

```bash
monitoring/
├── __init__.py
├── admin.py
├── apps.py
├── models.py                   # MonitoringEvent, PerformanceMetric
├── views.py                    # Dashboard views
├── api_views.py                # Monitoring API
├── serializers.py              # API serializers
├── urls.py
├── tests.py
├── migrations/
├── templates/
│   └── monitoring/
│       ├── dashboard.html
│       ├── events.html
│       └── metrics.html
├── static/
│   └── monitoring/
│       ├── css/
│       │   └── dashboard.css
│       └── js/
│           └── dashboard.js
└── dashboards/                 # Grafana dashboards
    └── mkt_dashboard.json
```

### Search Application (`search/`)

```bash
search/
├── __init__.py
├── apps.py
├── views.py                    # Search views
├── urls.py
├── tests.py
└── templates/
    └── search/
        └── search.html
```

## Project Configuration (`mkt/`)

```bash
mkt/
├── __init__.py
├── wsgi.py                     # WSGI configuration
├── asgi.py                     # ASGI configuration (future)
├── urls.py                     # Root URL configuration
└── settings/                   # Environment-specific settings
    ├── __init__.py
    ├── base.py                 # Base settings
    ├── development.py          # Development settings
    ├── production.py           # Production settings
    └── testing.py              # Test settings
```

## Templates Structure (`templates/`)

```bash
templates/
├── base.html                   # Base template
├── 404.html                    # Error pages
├── 500.html
├── includes/                   # Reusable components
│   ├── header.html
│   ├── footer.html
│   ├── navigation.html
│   └── meta.html
├── wagtailadmin/              # Wagtail admin customizations
│   └── base.html
└── registration/              # Authentication templates
    ├── login.html
    └── logout.html
```

## Static Files Structure (`static/`)

```bash
static/
├── css/
│   ├── base.css               # Global styles
│   ├── components.css         # Reusable components
│   └── utilities.css          # Utility classes
├── js/
│   ├── base.js                # Global JavaScript
│   ├── components.js          # Component scripts
│   └── vendor/                # Third-party libraries
│       ├── jquery.min.js
│       └── bootstrap.min.js
├── images/
│   ├── logo.png
│   ├── favicon.ico
│   └── default-og-image.jpg
├── fonts/                     # Custom fonts
└── admin/                     # Admin interface assets
```

## Configuration Files

### Environment Configuration (`.env.example`)

```bash
# Django Settings
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration
DATABASE_NAME=mkt-db
DATABASE_USER=postgres
DATABASE_PASSWORD=your-password
DATABASE_HOST=localhost
DATABASE_PORT=5432

# Monitoring Configuration
PROMETHEUS_METRICS_EXPORT_PORT=8001

# Email Configuration (Optional)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
EMAIL_USE_TLS=True

# Cache Configuration
CACHE_BACKEND=django.core.cache.backends.filebased.FileBasedCache
CACHE_LOCATION=/path/to/cache

# Security Settings (Production)
SECURE_SSL_REDIRECT=True
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=True
SECURE_HSTS_PRELOAD=True
```

### Docker Configuration (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: mkt-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - monitoring
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: mkt-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    networks:
      - monitoring
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:

networks:
  monitoring:
    driver: bridge
```

## Database Structure

### Models Organization

```python
# blog/models.py
class BlogCategory(models.Model):
    """Blog category snippet"""
    
class BlogAuthor(models.Model):
    """Blog author profile"""
    
class BlogIndexPage(RoutablePageMixin, Page):
    """Blog listing page with routing"""
    
class BlogPost(Page):
    """Individual blog post"""

# monitoring/models.py
class MonitoringEvent(models.Model):
    """Application monitoring events"""
    
class PerformanceMetric(models.Model):
    """Performance metrics storage"""

# home/models.py
class HomePage(Page):
    """Site homepage"""
    
class StandardPage(Page):
    """Generic content page"""
```

### Database Schema

```sql
-- Core Wagtail tables
wagtailcore_page
wagtailcore_site
wagtailimages_image
wagtaildocs_document

-- Blog tables
blog_blogcategory
blog_blogauthor
blog_blogpost
blog_blogpost_categories
blog_blogpost_tags

-- Monitoring tables
monitoring_monitoringevent
monitoring_performancemetric

-- Django auth tables
auth_user
auth_group
auth_permission
```

## API Structure

### URL Patterns

```python
# mkt/urls.py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('home.urls')),
    path('blog/', include('blog.urls')),
    path('monitoring/', include('monitoring.urls')),
    path('search/', include('search.urls')),
    path('api/v2/', api_router.urls),
]

# blog/urls.py
urlpatterns = [
    path('', BlogIndexPageView.as_view(), name='index'),
    path('api/', include([
        path('posts/', BlogPostListAPIView.as_view(), name='post_list_api'),
        path('posts/<slug:slug>/', BlogPostDetailAPIView.as_view(), name='post_detail_api'),
        path('categories/', BlogCategoryListAPIView.as_view(), name='category_list_api'),
        path('authors/', BlogAuthorListAPIView.as_view(), name='author_list_api'),
        path('stats/', blog_stats_api_view, name='stats_api'),
    ])),
]
```

### API Response Structure

```json
{
  "count": 25,
  "next": "http://localhost:8000/blog/api/posts/?page=2",
  "previous": null,
  "results": [
    {
      "id": 1,
      "title": "Blog Post Title",
      "slug": "blog-post-title",
      "excerpt": "Post excerpt...",
      "date": "2024-01-15T10:30:00Z",
      "author": {
        "id": 1,
        "name": "John Doe",
        "bio": "Author bio..."
      },
      "categories": [
        {
          "id": 1,
          "name": "Technology",
          "slug": "technology"
        }
      ],
      "tags_list": ["django", "python", "web"],
      "featured_image_url": "/media/images/featured.jpg",
      "estimated_reading_time": 5,
      "view_count": 150,
      "url": "/blog/blog-post-title/"
    }
  ]
}
```

## Testing Structure

### Test Organization

```python
# blog/tests.py
class BlogModelTests(TestCase):
    """Test blog models functionality"""
    
class BlogPageTests(WagtailPageTests):
    """Test blog page functionality"""
    
class BlogAPITests(APITestCase):
    """Test blog API endpoints"""

# monitoring/tests.py
class MonitoringModelTests(TestCase):
    """Test monitoring models"""
    
class MonitoringAPITests(APITestCase):
    """Test monitoring API"""
```

### Test Data Structure

```python
# Test fixtures
fixtures/
├── test_users.json
├── test_pages.json
├── test_blog_data.json
└── test_monitoring_data.json
```

## Deployment Structure

### Production Directory Layout

```bash
/var/www/mkt/
├── app/                        # Application code
├── static/                     # Collected static files
├── media/                      # User uploads
├── logs/                       # Application logs
├── backups/                    # Database backups
├── ssl/                        # SSL certificates
└── scripts/                    # Deployment scripts
    ├── deploy.sh
    ├── backup.sh
    └── restart.sh
```

### Configuration Management

```bash
config/
├── nginx/
│   ├── sites-available/
│   │   └── mkt.conf
│   └── ssl/
├── systemd/
│   ├── mkt.service
│   └── mkt-celery.service
├── supervisor/
│   └── mkt.conf
└── logrotate/
    └── mkt
```

## Development Workflow

### Git Branch Structure

```bash
main                           # Production branch
├── develop                    # Development branch
├── feature/blog-improvements  # Feature branches
├── feature/monitoring-alerts
├── hotfix/security-patch      # Hotfix branches
└── release/v1.1.0            # Release branches
```

### Development Commands

```bash
# Setup development environment
./setup.sh

# Run development server
python manage.py runserver

# Run tests
python manage.py test

# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic

# Create superuser
python manage.py createsuperuser

# Load sample data
python manage.py create_sample_data
```

## Code Organization Principles

### Separation of Concerns

- **Models**: Data structure and business logic
- **Views**: Request handling and response generation
- **Templates**: Presentation layer
- **Static Files**: Client-side assets
- **Tests**: Quality assurance

### Naming Conventions

- **Files**: lowercase with underscores (`blog_post.html`)
- **Classes**: PascalCase (`BlogPost`)
- **Functions/Variables**: snake_case (`get_blog_posts`)
- **Constants**: UPPER_CASE (`DEFAULT_PAGE_SIZE`)
- **URLs**: kebab-case (`/blog/my-post-title/`)

### Import Organization

```python
# Standard library imports
import os
import json
from datetime import datetime

# Third-party imports
from django.db import models
from wagtail.models import Page
from rest_framework import serializers

# Local application imports
from .models import BlogPost
from ..monitoring.models import MonitoringEvent
