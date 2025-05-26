from rest_framework import generics, filters
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count, Q
from .models import BlogPost, BlogCategory, BlogAuthor
from .serializers import (
    BlogPostListSerializer, 
    BlogPostDetailSerializer,
    BlogCategorySerializer,
    BlogAuthorSerializer
)
from django.db import models
import django_filters

import logging

logger = logging.getLogger(__name__)

class BlogPostFilter(django_filters.FilterSet):
    """Custom filter set for blog posts to handle tags properly."""
    
    tags = django_filters.CharFilter(field_name='tags__name', lookup_expr='exact')
    
    class Meta:
        model = BlogPost
        fields = ['categories', 'author']




class BlogPostDetailAPIView(generics.RetrieveAPIView):
    """API view for blog post detail."""
    
    serializer_class = BlogPostDetailSerializer
    lookup_field = 'slug'
    
    def get_queryset(self):
        """Get queryset with optimized queries."""
        return BlogPost.objects.live().public().select_related(
            'author', 'author__user'
        ).prefetch_related(
            'categories', 'tags', 'featured_image'
        )
    
    def retrieve(self, request, *args, **kwargs):
        """Override retrieve to increment view count."""
        instance = self.get_object()
        
        # Increment view count
        BlogPost.objects.filter(id=instance.id).update(
            view_count=models.F('view_count') + 1
        )
        
        serializer = self.get_serializer(instance)
        logger.info(f"Blog API: Retrieved post {instance.title}")
        return Response(serializer.data)

class BlogCategoryListAPIView(generics.ListAPIView):
    """API view for listing blog categories."""
    
    serializer_class = BlogCategorySerializer
    
    def get_queryset(self):
        """Get categories with post counts."""
        return BlogCategory.objects.annotate(
            post_count=Count('blogpost', filter=Q(blogpost__live=True))
        ).filter(post_count__gt=0)


class BlogAuthorListAPIView(generics.ListAPIView):
    """API view for listing blog authors."""
    
    serializer_class = BlogAuthorSerializer
    
    def get_queryset(self):
        """Get authors with published posts."""
        return BlogAuthor.objects.annotate(
            post_count=Count('blogpost', filter=Q(blogpost__live=True))
        ).filter(post_count__gt=0)


class BlogPostListAPIView(generics.ListAPIView):
    """API view for listing blog posts."""
    
    serializer_class = BlogPostListSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['categories__slug', 'author', 'tags__name']
    search_fields = ['title', 'excerpt', 'body']
    ordering_fields = ['date', 'view_count', 'title']
    ordering = ['-date']
    
    def get_queryset(self):
        """Get queryset with optimized queries."""
        queryset = BlogPost.objects.live().public().select_related(
            'author', 'author__user'
        ).prefetch_related(
            'categories', 'tags', 'featured_image'
        )
        
        # Filter by category slug if provided
        category_slug = self.request.query_params.get('category')
        if category_slug:
            queryset = queryset.filter(categories__slug=category_slug)
        
        # Filter by tag if provided
        tag = self.request.query_params.get('tag')
        if tag:
            queryset = queryset.filter(tags__name=tag)
        
        logger.info(f"Blog API: Retrieved {queryset.count()} posts")
        return queryset

@api_view(['GET'])
def blog_stats_api_view(request):
    """API view for blog statistics."""
    
    recent_posts = BlogPost.objects.live().public().order_by('-date')[:5]
    popular_posts = BlogPost.objects.live().public().order_by('-view_count')[:5]
    
    stats = {
        'total_posts': BlogPost.objects.live().public().count(),
        'total_categories': BlogCategory.objects.count(),
        'total_authors': BlogAuthor.objects.count(),
        'total_views': BlogPost.objects.live().public().aggregate(
            total=models.Sum('view_count')
        )['total'] or 0,
        'recent_posts': BlogPostListSerializer(
            recent_posts,
            many=True,
            context={'request': request}
        ).data,
        'popular_posts': BlogPostListSerializer(
            popular_posts,
            many=True,
            context={'request': request}
        ).data,
    }
    
    logger.info("Blog API: Retrieved blog statistics")
    return Response(stats)
