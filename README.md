# MKT Project

A modern Django-based web application built with Wagtail CMS, featuring a comprehensive blog system, real-time monitoring, and production-ready architecture.

## 🚀 Features

- **SEO-optimized Blog System** with categories, tags, and authors
- **Wagtail CMS** for content management
- **REST API** for headless content access
- **Real-time Monitoring** with Prometheus and Grafana
- **Performance Metrics** and event tracking
- **Responsive Design** with modern UI/UX
- **Production-ready** configuration

## 🛠 Technology Stack

- **Backend**: Django 5.2, Wagtail CMS
- **Database**: PostgreSQL 14+
- **API**: Django REST Framework
- **Monitoring**: Prometheus, Grafana, django-prometheus
- **Caching**: Django Cache Framework
- **Frontend**: HTML5, CSS3, JavaScript
- **Containerization**: Docker, Docker Compose

## 📋 Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Docker & Docker Compose (for monitoring)
- Git

## ⚡ Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/mkt-project.git
cd mkt-project
chmod +x setup.sh
./setup.sh
```

### 2. Configure Environment

Update `.env` file with your database credentials:

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Start Application

```bash
source venv/bin/activate
python manage.py runserver
```

### 4. Start Monitoring (Optional)

```bash
docker-compose up -d
```

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Django App** | [http://localhost:8000](http://localhost:8000) | Main application |
| **Admin Panel** | [http://localhost:8000/admin/](http://localhost:8000/admin/) | Wagtail admin |
| **Blog** | [http://localhost:8000/blog/](http://localhost:8000/blog/) | Blog section |
| **API** | [http://localhost:8000/blog/api/](http://localhost:8000/blog/api/) | REST API |
| **Monitoring** | [http://localhost:8000/monitoring/dashboard/](http://localhost:8000/monitoring/dashboard/) | Django monitoring |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Metrics collection |
| **Grafana** | [http://localhost:3000](http://localhost:3000) | Metrics visualization |

## 📊 API Reference

### Blog API Endpoints

```bash
# List all blog posts
GET /blog/api/posts/

# Filter by category
GET /blog/api/posts/?category=technology

# Filter by tag
GET /blog/api/posts/?tag=django

# Get specific post
GET /blog/api/posts/{slug}/

# List categories
GET /blog/api/categories/

# List authors
GET /blog/api/authors/

# Blog statistics
GET /blog/api/stats/
```

### Monitoring API Endpoints

```bash
# List monitoring events
GET /monitoring/api/events/

# Create monitoring event
POST /monitoring/api/events/

# Performance metrics
GET /monitoring/api/metrics/

# Monitoring statistics
GET /monitoring/api/stats/
```

## 🏗 Project Structure

```bash
mkt-project/
├── blog/                       # Blog application
│   ├── models.py              # Blog models (Post, Category, Author)
│   ├── views.py               # Blog views
│   ├── api_views.py           # API views
│   ├── serializers.py         # DRF serializers
│   └── templates/             # Blog templates
├── home/                       # Home page application
├── monitoring/                 # Monitoring application
│   ├── models.py              # Monitoring models
│   ├── views.py               # Monitoring dashboard
│   ├── api_views.py           # Monitoring API
│   └── dashboards/            # Grafana dashboards
├── mkt/                        # Project configuration
│   ├── settings/              # Environment-specific settings
│   ├── urls.py                # URL configuration
│   └── wsgi.py                # WSGI configuration
├── prometheus/                 # Prometheus configuration
├── grafana/                    # Grafana configuration
├── templates/                  # Global templates
├── static/                     # Static files
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Container orchestration
└── setup.sh                   # Setup script
```

## 🔧 Development

### Running Tests

```bash
python manage.py test
```

### Creating Blog Content

1. Access admin panel: [http://localhost:8000/admin/](http://localhost:8000/admin/)
2. Create blog categories and authors
3. Create blog index page under Home
4. Add blog posts under the blog index

### Adding Custom Monitoring

```python
from monitoring.models import MonitoringEvent

# Log custom events
MonitoringEvent.objects.create(
    event_type='info',
    source='my_module',
    message='Custom event',
    details={'key': 'value'}
)
```

### Custom Metrics

```python
from prometheus_client import Counter

# Define custom metrics
my_counter = Counter('my_custom_metric', 'Description')
my_counter.inc()
```

## 📈 Monitoring & Analytics

### Django Monitoring Dashboard

- Real-time event tracking
- Performance metrics
- System health monitoring
- Custom event logging

### Prometheus Metrics

- HTTP request metrics
- Database query performance
- Cache hit/miss ratios
- Custom application metrics

### Grafana Dashboards

- Real-time visualizations
- Performance monitoring
- Alert configuration
- Custom dashboard creation

## 🚀 Deployment

### Production Settings

1. Set environment variables:

    ```bash
    DEBUG=False
    SECRET_KEY=your-production-secret-key
    ALLOWED_HOSTS=yourdomain.com
    ```

2. Configure database and static files
3. Set up reverse proxy (Nginx)
4. Configure SSL certificates
5. Set up monitoring and logging

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key | Required |
| `DEBUG` | Debug mode | `False` |
| `ALLOWED_HOSTS` | Allowed hosts | Required |
| `DATABASE_URL` | Database connection | Required |
| `PROMETHEUS_METRICS_EXPORT_PORT` | Metrics port | `8001` |

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Architecture Overview](docs/architecture.md)
- [Site Structure](docs/site_structure.md)
- [Project Structure](docs/project_structure.md)
- [Design Practices](docs/design_practices.md)
- [Deployment Guide](docs/deployment.md)

## 🧪 Testing

```bash
# Run all tests
python manage.py test

# Run specific app tests
python manage.py test blog
python manage.py test monitoring

# Run with coverage
coverage run --source='.' manage.py test
coverage report
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `python manage.py test`
6. Commit changes: `git commit -m 'Add feature'`
7. Push to branch: `git push origin feature-name`
8. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `docs/` directory
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Advanced search functionality
- [ ] Email newsletter integration
- [ ] Social media integration
- [ ] Advanced analytics
- [ ] Mobile app API
- [ ] Kubernetes deployment

## Built with ❤️ using Django and Wagtail**
