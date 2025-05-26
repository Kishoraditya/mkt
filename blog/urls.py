from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views, api_views

app_name = 'blog'

# API URLs
api_urlpatterns = [
    path('posts/', api_views.BlogPostListAPIView.as_view(), name='post_list_api'),
    path('posts/<slug:slug>/', api_views.BlogPostDetailAPIView.as_view(), name='post_detail_api'),
    path('categories/', api_views.BlogCategoryListAPIView.as_view(), name='category_list_api'),
    path('authors/', api_views.BlogAuthorListAPIView.as_view(), name='author_list_api'),
    path('stats/', api_views.blog_stats_api_view, name='blog_stats_api'),
]

urlpatterns = [
    path('api/', include(api_urlpatterns)),
]