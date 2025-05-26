# Deployment Guide

## Overview

This guide covers deploying the MKT project to various environments, from development to production, including containerized deployments and cloud platforms.

## Deployment Environments

### Development Environment

#### Local Development Setup

```bash
# Clone repository
git clone https://github.com/kishoraditya/mkt-project.git
cd mkt-project

# Setup environment
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Start development server
python manage.py runserver

# Start monitoring (optional)
docker-compose up -d prometheus grafana
```

#### Development Configuration

```python
# mkt/settings/development.py
from .base import *

DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mkt_dev',
        'USER': 'postgres',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# Static files
STATIC_URL = '/static/'
MEDIA_URL = '/media/'

# Email backend for development
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Disable caching in development
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
    }
}
```

### Staging Environment

#### Staging Configuration

```python
# mkt/settings/staging.py
from .base import *

DEBUG = False
ALLOWED_HOSTS = ['staging.mkt-project.com']

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DATABASE_NAME'),
        'USER': config('DATABASE_USER'),
        'PASSWORD': config('DATABASE_PASSWORD'),
        'HOST': config('DATABASE_HOST'),
        'PORT': config('DATABASE_PORT'),
        'OPTIONS': {
            'sslmode': 'require',
        },
    }
}

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = '/var/www/mkt/static/'
MEDIA_URL = '/media/'
MEDIA_ROOT = '/var/www/mkt/media/'

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'

# Email configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = config('EMAIL_HOST')
EMAIL_PORT = config('EMAIL_PORT', cast=int)
EMAIL_USE_TLS = config('EMAIL_USE_TLS', cast=bool)
EMAIL_HOST_USER = config('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD')

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/mkt/django.log',
            'maxBytes': 1024*1024*15,  # 15MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
        'console': {
            'level': 'ERROR',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO',
    },
}
```

### Production Environment

#### Production Configuration

```python
# mkt/settings/production.py
from .base import *
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

DEBUG = False
ALLOWED_HOSTS = ['mkt-project.com', 'www.mkt-project.com']

# Database with connection pooling
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DATABASE_NAME'),
        'USER': config('DATABASE_USER'),
        'PASSWORD': config('DATABASE_PASSWORD'),
        'HOST': config('DATABASE_HOST'),
        'PORT': config('DATABASE_PORT'),
        'OPTIONS': {
            'sslmode': 'require',
            'MAX_CONNS': 20,
            'OPTIONS': {
                'MAX_CONNS': 20,
            }
        },
        'CONN_MAX_AGE': 600,
    }
}

# Static files with CDN
STATIC_URL = 'https://cdn.mkt-project.com/static/'
STATIC_ROOT = '/var/www/mkt/static/'
MEDIA_URL = 'https://cdn.mkt-project.com/media/'
MEDIA_ROOT = '/var/www/mkt/media/'

# AWS S3 Configuration (optional)
if config('USE_S3', default=False, cast=bool):
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    STATICFILES_STORAGE = 'storages.backends.s3boto3.StaticS3Boto3Storage'
    AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = config('AWS_S3_REGION_NAME')
    AWS_S3_CUSTOM_DOMAIN = config('AWS_S3_CUSTOM_DOMAIN')

# Redis Cache
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': config('REDIS_URL', default='redis://127.0.0.1:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Session storage
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'
X_FRAME_OPTIONS = 'DENY'

# Content Security Policy
CSP_DEFAULT_SRC = ("'self'",)
CSP_SCRIPT_SRC = ("'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net")
CSP_STYLE_SRC = ("'self'", "'unsafe-inline'", "https://fonts.googleapis.com")
CSP_FONT_SRC = ("'self'", "https://fonts.gstatic.com")
CSP_IMG_SRC = ("'self'", "data:", "https:")

# Error tracking with Sentry
sentry_sdk.init(
    dsn=config('SENTRY_DSN', default=''),
    integrations=[
        DjangoIntegration(),
        LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
    ],
    traces_sample_rate=0.1,
    send_default_pii=True,
    environment='production',
)

# Email configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = config('EMAIL_HOST')
EMAIL_PORT = config('EMAIL_PORT', cast=int)
EMAIL_USE_TLS = config('EMAIL_USE_TLS', cast=bool)
EMAIL_HOST_USER = config('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD')
DEFAULT_FROM_EMAIL = config('DEFAULT_FROM_EMAIL')
SERVER_EMAIL = config('SERVER_EMAIL')

# Celery configuration for background tasks
CELERY_BROKER_URL = config('CELERY_BROKER_URL', default='redis://127.0.0.1:6379/0')
CELERY_RESULT_BACKEND = config('CELERY_RESULT_BACKEND', default='redis://127.0.0.1:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE
```

## Server Configuration

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/mkt-project.com
upstream mkt_app {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name mkt-project.com www.mkt-project.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name mkt-project.com www.mkt-project.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/mkt-project.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mkt-project.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Static files
    location /static/ {
        alias /var/www/mkt/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
        
        # Brotli compression
        location ~* \.(js|css)$ {
            add_header Content-Encoding br;
            add_header Vary Accept-Encoding;
        }
    }

    # Media files
    location /media/ {
        alias /var/www/mkt/media/;
        expires 1M;
        add_header Cache-Control "public";
    }

    # Favicon
    location = /favicon.ico {
        alias /var/www/mkt/static/images/favicon.ico;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Robots.txt
    location = /robots.txt {
        alias /var/www/mkt/static/robots.txt;
        expires 1d;
    }

    # Django application
    location / {
        proxy_pass http://mkt_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    # Health check endpoint
    location /health/ {
        access_log off;
        proxy_pass http://mkt_app;
        proxy_set_header Host $host;
    }

    # Rate limiting for API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://mkt_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Rate limiting zones
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
}
```

### Gunicorn Configuration

```python
# gunicorn.conf.py
import multiprocessing

# Server socket
bind = "127.0.0.1:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/var/log/mkt/gunicorn_access.log"
errorlog = "/var/log/mkt/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "mkt_gunicorn"

# Server mechanics
daemon = False
pidfile = "/var/run/mkt/gunicorn.pid"
user = "mkt"
group = "mkt"
tmp_upload_dir = None

# SSL (if terminating SSL at Gunicorn)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Preload application
preload_app = True

# Worker lifecycle hooks
def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")
```

### Systemd Service Configuration

```ini
# /etc/systemd/system/mkt.service
[Unit]
Description=MKT Django Application
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=notify
User=mkt
Group=mkt
WorkingDirectory=/var/www/mkt
Environment=DJANGO_SETTINGS_MODULE=mkt.settings.production
ExecStart=/var/www/mkt/venv/bin/gunicorn --config /var/www/mkt/gunicorn.conf.py mkt.wsgi:application
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/mkt-celery.service
[Unit]
Description=MKT Celery Worker
After=network.target redis.service
Requires=redis.service

[Service]
Type=forking
User=mkt
Group=mkt
WorkingDirectory=/var/www/mkt
Environment=DJANGO_SETTINGS_MODULE=mkt.settings.production
ExecStart=/var/www/mkt/venv/bin/celery -A mkt worker --loglevel=info --pidfile=/var/run/mkt/celery.pid
ExecStop=/bin/kill -s TERM $MAINPID
ExecReload=/bin/kill -s HUP $MAINPID
PIDFile=/var/run/mkt/celery.pid
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Docker Deployment

### Production Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user
RUN groupadd -r mkt && useradd -r -g mkt mkt

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/static /app/media /app/logs /var/log/mkt /var/run/mkt && \
    chown -R mkt:mkt /app /var/log/mkt /var/run/mkt

# Copy configuration files
COPY docker/nginx.conf /etc/nginx/sites-available/default
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Collect static files
RUN python manage.py collectstatic --noinput --settings=mkt.settings.production

# Change ownership
RUN chown -R mkt:mkt /app

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health/ || exit 1

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
```

### Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: ${DATABASE_NAME}
      POSTGRES_USER: ${DATABASE_USER}
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - mkt_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DATABASE_USER}"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    networks:
      - mkt_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  web:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    environment:
      - DJANGO_SETTINGS_MODULE=mkt.settings.production
      - DATABASE_HOST=db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - static_volume:/app/static
      - media_volume:/app/media
      - ./logs:/app/logs
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - mkt_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health/"]
      interval: 30s
      timeout: 10s
      retries: 3

  celery:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    command: celery -A mkt worker --loglevel=info
    environment:
      - DJANGO_SETTINGS_MODULE=mkt.settings.production
      - DATABASE_HOST=db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    networks:
      - mkt_network
    restart: unless-stopped

  celery-beat:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    command: celery -A mkt beat --loglevel=info --scheduler django_celery_beat.schedulers:DatabaseScheduler
    environment:
      - DJANGO_SETTINGS_MODULE=mkt.settings.production
      - DATABASE_HOST=db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    networks:
      - mkt_network
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - mkt_network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    networks:
      - mkt_network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  static_volume:
  media_volume:
  prometheus_data:
  grafana_data:

networks:
  mkt_network:
    driver: bridge
```

### Docker Configuration Files

```ini
# docker/supervisord.conf
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

[program:gunicorn]
command=/opt/venv/bin/gunicorn --config /app/gunicorn.conf.py mkt.wsgi:application
directory=/app
user=mkt
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/mkt/gunicorn.log

[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/nginx/access.log
stderr_logfile=/var/log/nginx/error.log

[program:migrate]
command=/opt/venv/bin/python manage.py migrate --noinput
directory=/app
user=mkt
autostart=true
autorestart=false
startsecs=0
redirect_stderr=true
stdout_logfile=/var/log/mkt/migrate.log
```

```nginx
# docker/nginx.conf
upstream django {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name _;
    client_max_body_size 100M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Static files
    location /static/ {
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Media files
    location /media/ {
        alias /app/media/;
        expires 1M;
        add_header Cache-Control "public";
    }

    # Health check
    location /health/ {
        access_log off;
        proxy_pass http://django;
        proxy_set_header Host $host;
    }

    # Main application
    location / {
        proxy_pass http://django;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
    }
}
```

## Cloud Platform Deployments

### AWS Deployment with Elastic Beanstalk

```yaml
# .ebextensions/01_packages.config
packages:
  yum:
    postgresql-devel: []
    libjpeg-turbo-devel: []
    libpng-devel: []
    freetype-devel: []

option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: mkt.wsgi:application
  aws:elasticbeanstalk:application:environment:
    DJANGO_SETTINGS_MODULE: mkt.settings.production
    PYTHONPATH: /opt/python/current/app
  aws:elasticbeanstalk:container:python:staticfiles:
    /static/: static/
```

```yaml
# .ebextensions/02_django.config
container_commands:
  01_migrate:
    command: "source /opt/python/run/venv/bin/activate && python manage.py migrate --noinput"
    leader_only: true
  02_collectstatic:
    command: "source /opt/python/run/venv/bin/activate && python manage.py collectstatic --noinput"
  03_createsu:
    command: "source /opt/python/run/venv/bin/activate && python manage.py createsu"
    leader_only: true

option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: mkt/wsgi.py
```

### Heroku Deployment

```python
# Procfile
web: gunicorn mkt.wsgi:application --log-file -
worker: celery -A mkt worker --loglevel=info
beat: celery -A mkt beat --loglevel=info
```

```python
# runtime.txt
python-3.11.0
```

```bash
# Heroku deployment script
#!/bin/bash

# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login to Heroku
heroku login

# Create Heroku app
heroku create mkt-project

# Add buildpacks
heroku buildpacks:add heroku/python

# Add PostgreSQL addon
heroku addons:create heroku-postgresql:hobby-dev

# Add Redis addon
heroku addons:create heroku-redis:hobby-dev

# Set environment variables
heroku config:set DJANGO_SETTINGS_MODULE=mkt.settings.production
heroku config:set SECRET_KEY=$(python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())')
heroku config:set DEBUG=False
heroku config:set ALLOWED_HOSTS=mkt-project.herokuapp.com

# Deploy
git push heroku main

# Run migrations
heroku run python manage.py migrate

# Create superuser
heroku run python manage.py createsuperuser

# Scale dynos
heroku ps:scale web=1 worker=1
```

### DigitalOcean App Platform

```yaml
# .do/app.yaml
name: mkt-project
services:
- name: web
  source_dir: /
  github:
    repo: yourusername/mkt-project
    branch: main
  run_command: gunicorn --worker-tmp-dir /dev/shm mkt.wsgi:application
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: DJANGO_SETTINGS_MODULE
    value: mkt.settings.production
  - key: SECRET_KEY
    value: ${SECRET_KEY}
  - key: DEBUG
    value: "False"
  - key: DATABASE_URL
    value: ${db.DATABASE_URL}
  - key: REDIS_URL
    value: ${redis.REDIS_URL}

- name: worker
  source_dir: /
  github:
    repo: yourusername/mkt-project
    branch: main
  run_command: celery -A mkt worker --loglevel=info
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: DJANGO_SETTINGS_MODULE
    value: mkt.settings.production
  - key: DATABASE_URL
    value: ${db.DATABASE_URL}
  - key: REDIS_URL
    value: ${redis.REDIS_URL}

databases:
- name: db
  engine: PG
  num_nodes: 1
  size: db-s-dev-database
  version: "13"

- name: redis
  engine: REDIS
  num_nodes: 1
  size: db-s-dev-database
  version: "6"

static_sites:
- name: static
  source_dir: /static
  github:
    repo: yourusername/mkt-project
    branch: main
  build_command: python manage.py collectstatic --noinput
```

## Deployment Scripts

### Automated Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
PROJECT_NAME="mkt"
PROJECT_DIR="/var/www/$PROJECT_NAME"
REPO_URL="https://github.com/yourusername/mkt-project.git"
BRANCH="main"
VENV_DIR="$PROJECT_DIR/venv"
BACKUP_DIR="/var/backups/$PROJECT_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting deployment of $PROJECT_NAME${NC}"

# Create backup
echo -e "${YELLOW}Creating backup...${NC}"
mkdir -p $BACKUP_DIR
sudo -u postgres pg_dump $PROJECT_NAME > $BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S).sql

# Pull latest code
echo -e "${YELLOW}Pulling latest code...${NC}"
cd $PROJECT_DIR
git fetch origin
git reset --hard origin/$BRANCH

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source $VENV_DIR/bin/activate

# Install/update dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Run migrations
echo -e "${YELLOW}Running migrations...${NC}"
python manage.py migrate --noinput

# Collect static files
echo -e "${YELLOW}Collecting static files...${NC}"
python manage.py collectstatic --noinput

# Restart services
echo -e "${YELLOW}Restarting services...${NC}"
sudo systemctl restart $PROJECT_NAME
sudo systemctl restart $PROJECT_NAME-celery
sudo systemctl reload nginx

# Health check
echo -e "${YELLOW}Performing health check...${NC}"
sleep 5
if curl -f http://localhost/health/ > /dev/null 2>&1; then
    echo -e "${GREEN}Deployment successful!${NC}"
else
    echo -e "${RED}Health check failed! Rolling back...${NC}"
    # Rollback logic here
    exit 1
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"
```

### Zero-Downtime Deployment Script

```bash
#!/bin/bash
# zero_downtime_deploy.sh

set -e

PROJECT_NAME="mkt"
PROJECT_DIR="/var/www/$PROJECT_NAME"
CURRENT_DIR="$PROJECT_DIR/current"
RELEASES_DIR="$PROJECT_DIR/releases"
SHARED_DIR="$PROJECT_DIR/shared"
RELEASE_DIR="$RELEASES_DIR/$(date +%Y%m%d_%H%M%S)"

echo "Starting zero-downtime deployment..."

# Create release directory
```bash
# Create release directory
mkdir -p $RELEASE_DIR

# Clone code to release directory
echo "Cloning code to release directory..."
git clone --depth 1 --branch main $REPO_URL $RELEASE_DIR

# Create shared directories if they don't exist
mkdir -p $SHARED_DIR/{media,logs,static}

# Link shared directories
echo "Linking shared directories..."
ln -nfs $SHARED_DIR/media $RELEASE_DIR/media
ln -nfs $SHARED_DIR/logs $RELEASE_DIR/logs
ln -nfs $SHARED_DIR/static $RELEASE_DIR/static

# Copy environment file
cp $SHARED_DIR/.env $RELEASE_DIR/.env

# Create virtual environment for this release
echo "Setting up virtual environment..."
python3 -m venv $RELEASE_DIR/venv
source $RELEASE_DIR/venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r $RELEASE_DIR/requirements.txt

# Run migrations
echo "Running migrations..."
cd $RELEASE_DIR
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Test the new release
echo "Testing new release..."
python manage.py check --deploy

# Switch to new release
echo "Switching to new release..."
ln -nfs $RELEASE_DIR $CURRENT_DIR

# Reload application servers
echo "Reloading application servers..."
sudo systemctl reload $PROJECT_NAME

# Health check
echo "Performing health check..."
sleep 5
for i in {1..5}; do
    if curl -f http://localhost/health/ > /dev/null 2>&1; then
        echo "Health check passed!"
        break
    else
        echo "Health check attempt $i failed, retrying..."
        sleep 2
    fi
done

# Clean up old releases (keep last 5)
echo "Cleaning up old releases..."
cd $RELEASES_DIR
ls -t | tail -n +6 | xargs rm -rf

echo "Zero-downtime deployment completed successfully!"
```

## Monitoring and Logging in Production

### Log Aggregation with ELK Stack

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - logging

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - ./logs:/var/log/mkt
    ports:
      - "5044:5044"
    environment:
      LS_JAVA_OPTS: "-Xmx256m -Xms256m"
    networks:
      - logging
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    networks:
      - logging
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:

networks:
  logging:
    driver: bridge
```

```ruby
# logstash/pipeline/logstash.conf
input {
  file {
    path => "/var/log/mkt/django.log"
    start_position => "beginning"
    type => "django"
  }
  file {
    path => "/var/log/mkt/gunicorn_access.log"
    start_position => "beginning"
    type => "gunicorn_access"
  }
  file {
    path => "/var/log/mkt/gunicorn_error.log"
    start_position => "beginning"
    type => "gunicorn_error"
  }
}

filter {
  if [type] == "django" {
    grok {
      match => { "message" => "%{LOGLEVEL:level} %{TIMESTAMP_ISO8601:timestamp} %{WORD:module} %{NUMBER:process:int} %{NUMBER:thread:int} %{GREEDYDATA:message}" }
    }
    date {
      match => [ "timestamp", "yyyy-MM-dd HH:mm:ss,SSS" ]
    }
  }
  
  if [type] == "gunicorn_access" {
    grok {
      match => { "message" => "%{IPORHOST:remote_addr} %{USER:remote_user} %{USER:authenticated_user} \[%{HTTPDATE:timestamp}\] \"%{WORD:method} %{URIPATHPARAM:request} HTTP/%{NUMBER:http_version}\" %{NUMBER:status:int} %{NUMBER:bytes:int} \"%{DATA:referrer}\" \"%{DATA:user_agent}\" %{NUMBER:request_time:int}" }
    }
    date {
      match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "mkt-logs-%{+YYYY.MM.dd}"
  }
  stdout { codec => rubydebug }
}
```

### Application Performance Monitoring

```python
# monitoring/apm.py
import time
import logging
from functools import wraps
from django.core.cache import cache
from django.db import connection
from django.conf import settings

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Application Performance Monitoring"""
    
    @staticmethod
    def monitor_view(view_func):
        """Decorator to monitor view performance"""
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            start_time = time.time()
            initial_queries = len(connection.queries)
            
            try:
                response = view_func(request, *args, **kwargs)
                status_code = response.status_code
            except Exception as e:
                status_code = 500
                logger.error(f"View {view_func.__name__} failed: {str(e)}")
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time
                query_count = len(connection.queries) - initial_queries
                
                # Log performance metrics
                logger.info(
                    f"View: {view_func.__name__}, "
                    f"Duration: {duration:.3f}s, "
                    f"Queries: {query_count}, "
                    f"Status: {status_code}"
                )
                
                # Store metrics for monitoring
                cache.set(
                    f"perf:{view_func.__name__}:{int(time.time())}",
                    {
                        'duration': duration,
                        'query_count': query_count,
                        'status_code': status_code,
                        'timestamp': time.time()
                    },
                    timeout=3600
                )
            
            return response
        return wrapper
    
    @staticmethod
    def monitor_database():
        """Monitor database performance"""
        from django.db import connections
        
        db_stats = {}
        for alias in connections:
            conn = connections[alias]
            db_stats[alias] = {
                'queries_count': len(conn.queries),
                'total_time': sum(float(q['time']) for q in conn.queries)
            }
        
        return db_stats
    
    @staticmethod
    def monitor_cache():
        """Monitor cache performance"""
        try:
            # Test cache connectivity
            cache.set('health_check', 'ok', 30)
            cache_status = cache.get('health_check') == 'ok'
            
            return {
                'status': 'healthy' if cache_status else 'unhealthy',
                'backend': settings.CACHES['default']['BACKEND']
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
```

### Health Check Endpoints

```python
# monitoring/health.py
import psutil
import redis
from django.http import JsonResponse
from django.db import connection
from django.core.cache import cache
from django.conf import settings

def health_check(request):
    """Comprehensive health check endpoint"""
    health_status = {
        'status': 'healthy',
        'timestamp': time.time(),
        'checks': {}
    }
    
    # Database check
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        health_status['checks']['database'] = {'status': 'healthy'}
    except Exception as e:
        health_status['checks']['database'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
        health_status['status'] = 'unhealthy'
    
    # Cache check
    try:
        cache.set('health_check', 'ok', 30)
        if cache.get('health_check') == 'ok':
            health_status['checks']['cache'] = {'status': 'healthy'}
        else:
            raise Exception("Cache test failed")
    except Exception as e:
        health_status['checks']['cache'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
        health_status['status'] = 'unhealthy'
    
    # System resources check
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status['checks']['system'] = {
            'status': 'healthy',
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent
        }
        
        # Alert if resources are high
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            health_status['checks']['system']['status'] = 'warning'
            
    except Exception as e:
        health_status['checks']['system'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Return appropriate HTTP status
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return JsonResponse(health_status, status=status_code)

def readiness_check(request):
    """Kubernetes readiness probe"""
    try:
        # Quick database check
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        return JsonResponse({'status': 'ready'})
    except Exception:
        return JsonResponse({'status': 'not ready'}, status=503)

def liveness_check(request):
    """Kubernetes liveness probe"""
    return JsonResponse({'status': 'alive'})
```

## Backup and Recovery

### Database Backup Script

```bash
#!/bin/bash
# backup_database.sh

set -e

# Configuration
DB_NAME="mkt_production"
DB_USER="mkt_user"
BACKUP_DIR="/var/backups/mkt"
RETENTION_DAYS=30
S3_BUCKET="mkt-backups"

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate backup filename
BACKUP_FILE="$BACKUP_DIR/mkt_backup_$(date +%Y%m%d_%H%M%S).sql"

echo "Starting database backup..."

# Create database backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE
BACKUP_FILE="$BACKUP_FILE.gz"

echo "Backup created: $BACKUP_FILE"

# Upload to S3 (if configured)
if command -v aws &> /dev/null && [ ! -z "$S3_BUCKET" ]; then
    echo "Uploading backup to S3..."
    aws s3 cp $BACKUP_FILE s3://$S3_BUCKET/database/
    echo "Backup uploaded to S3"
fi

# Clean up old backups
echo "Cleaning up old backups..."
find $BACKUP_DIR -name "mkt_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete

echo "Database backup completed successfully"
```

### Media Files Backup

```bash
#!/bin/bash
# backup_media.sh

set -e

MEDIA_DIR="/var/www/mkt/media"
BACKUP_DIR="/var/backups/mkt/media"
S3_BUCKET="mkt-backups"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create media backup
BACKUP_FILE="$BACKUP_DIR/media_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "Starting media backup..."
tar -czf $BACKUP_FILE -C $MEDIA_DIR .

echo "Media backup created: $BACKUP_FILE"

# Sync to S3
if command -v aws &> /dev/null && [ ! -z "$S3_BUCKET" ]; then
    echo "Syncing media to S3..."
    aws s3 sync $MEDIA_DIR s3://$S3_BUCKET/media/ --delete
    echo "Media synced to S3"
fi

echo "Media backup completed successfully"
```

### Automated Backup with Cron

```bash
# Add to crontab: crontab -e

# Database backup every 6 hours
0 */6 * * * /var/www/mkt/scripts/backup_database.sh >> /var/log/mkt/backup.log 2>&1

# Media backup daily at 2 AM
0 2 * * * /var/www/mkt/scripts/backup_media.sh >> /var/log/mkt/backup.log 2>&1

# Log rotation weekly
0 0 * * 0 /usr/sbin/logrotate /etc/logrotate.d/mkt
