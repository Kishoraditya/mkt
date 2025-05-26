from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework.test import APITestCase
from rest_framework import status
from .models import MonitoringEvent, PerformanceMetric
import json


class MonitoringModelTests(TestCase):
    """Test monitoring models."""
    
    def setUp(self):
        """Set up test data."""
        # Create test events
        self.event = MonitoringEvent.objects.create(
            event_type='error',
            source='test',
            message='Test error message',
            details={'code': 500}
        )
        
        # Create test metrics
        self.metric = PerformanceMetric.objects.create(
            name='response_time',
            value=0.5,
            unit='seconds',
            labels={'endpoint': '/api/blog/'}
        )
    
    def test_monitoring_event_str(self):
        """Test MonitoringEvent string representation."""
        self.assertIn('ERROR: test', str(self.event))
    
    def test_performance_metric_str(self):
        """Test PerformanceMetric string representation."""
        self.assertIn('response_time: 0.5 seconds', str(self.metric))


class MonitoringViewTests(TestCase):
    """Test monitoring views."""
    
    def setUp(self):
        """Set up test data."""
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
        # Create test client
        self.client = Client()
        
        # Create test events
        for i in range(5):
            MonitoringEvent.objects.create(
                event_type='info',
                source=f'test-{i}',
                message=f'Test message {i}'
            )
        
        # Create test metrics
        for i in range(5):
            PerformanceMetric.objects.create(
                name='cpu_usage',
                value=i * 10,
                unit='percent'
            )
    
    def test_dashboard_view_authenticated(self):
        """Test dashboard view with authenticated user."""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(reverse('monitoring:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Recent Events')
        self.assertContains(response, 'Event Counts')
    
    def test_dashboard_view_unauthenticated(self):
        """Test dashboard view with unauthenticated user."""
        response = self.client.get(reverse('monitoring:dashboard'))
        self.assertNotEqual(response.status_code, 200)


class MonitoringAPITests(APITestCase):
    """Test monitoring API endpoints."""
    
    def setUp(self):
        """Set up test data."""
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
        # Create test events
        for i in range(5):
            MonitoringEvent.objects.create(
                event_type='info' if i % 2 == 0 else 'error',
                source=f'test-{i}',
                message=f'Test message {i}',
                details={'code': i * 100}
            )
        
        # Create test metrics
        for i in range(5):
            PerformanceMetric.objects.create(
                name='cpu_usage' if i % 2 == 0 else 'memory_usage',
                value=i * 10,
                unit='percent' if i % 2 == 0 else 'MB',
                labels={'server': f'server-{i}'}
            )
    
    def test_events_list_authenticated(self):
        """Test events list API with authenticated user."""
        self.client.force_authenticate(user=self.user)
        response = self.client.get(reverse('monitoring:monitoringevent-list'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['results']), 5)
        self.assertIn('event_type', data['results'][0])
        self.assertIn('message', data['results'][0])
    
    def test_events_list_unauthenticated(self):
        """Test events list API with unauthenticated user."""
        response = self.client.get(reverse('monitoring:monitoringevent-list'))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_metrics_list_authenticated(self):
        """Test metrics list API with authenticated user."""
        self.client.force_authenticate(user=self.user)
        response = self.client.get(reverse('monitoring:performancemetric-list'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['results']), 5)
        self.assertIn('name', data['results'][0])
        self.assertIn('value', data['results'][0])
    
    def test_create_event(self):
        """Test creating a monitoring event."""
        self.client.force_authenticate(user=self.user)
        event_data = {
            'event_type': 'warning',
            'source': 'api-test',
            'message': 'Test warning message',
            'details': {'code': 400}
        }
        response = self.client.post(
            reverse('monitoring:monitoringevent-list'),
            event_data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(MonitoringEvent.objects.count(), 6)
        self.assertEqual(MonitoringEvent.objects.last().message, 'Test warning message')
    
    def test_create_metric(self):
        """Test creating a performance metric."""
        self.client.force_authenticate(user=self.user)
        metric_data = {
            'name': 'request_time',
            'value': 0.75,
            'unit': 'seconds',
            'labels': {'endpoint': '/api/blog/'}
        }
        response = self.client.post(
            reverse('monitoring:performancemetric-list'),
            metric_data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(PerformanceMetric.objects.count(), 6)
        self.assertEqual(PerformanceMetric.objects.last().name, 'request_time')
    
    def test_monitoring_stats(self):
        """Test monitoring stats API."""
        self.client.force_authenticate(user=self.user)
        response = self.client.get(reverse('monitoring:stats'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('total_events', data)
        self.assertIn('event_stats', data)
        self.assertIn('total_metrics', data)
        self.assertIn('metric_stats', data)
        self.assertEqual(data['total_events'], 5)
        self.assertEqual(data['total_metrics'], 5)
    
    def test_log_event(self):
        """Test log event API."""
        self.client.force_authenticate(user=self.user)
        event_data = {
            'event_type': 'debug',
            'source': 'test-api',
            'message': 'Debug message from API test',
            'details': {'test_id': 123}
        }
        response = self.client.post(
            reverse('monitoring:log_event'),
            event_data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(MonitoringEvent.objects.count(), 6)
        latest_event = MonitoringEvent.objects.latest('timestamp')
        self.assertEqual(latest_event.event_type, 'debug')
        self.assertEqual(latest_event.message, 'Debug message from API test')


class PrometheusMiddlewareTests(TestCase):
    """Test Prometheus middleware."""
    
    def setUp(self):
        """Set up test data."""
        self.client = Client()
    
    def test_middleware_metrics_collection(self):
        """Test that middleware collects metrics."""
        # This is a basic test to ensure the middleware doesn't break
        # Actual metric collection would be tested in integration tests
        response = self.client.get(reverse('home:home'))
        self.assertEqual(response.status_code, 200)
