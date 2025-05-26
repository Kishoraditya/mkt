from rest_framework import serializers
from wagtail.api.v2.serializers import PageSerializer
from .models import BlogPost, BlogCategory, BlogAuthor


class BlogAuthorSerializer(serializers.ModelSerializer):
    """Serializer for blog authors."""
    
    name = serializers.CharField(read_only=True)
    
    class Meta:
        model = BlogAuthor
        fields = ['id', 'name', 'bio', 'website', 'twitter', 'linkedin']


class BlogCategorySerializer(serializers.ModelSerializer):
    """Serializer for blog categories."""
    
    class Meta:
        model = BlogCategory
        fields = ['id', 'name', 'slug', 'description']


class BlogPostListSerializer(serializers.ModelSerializer):
    """Serializer for blog post list view."""
    
    author = BlogAuthorSerializer(read_only=True)
    categories = BlogCategorySerializer(many=True, read_only=True)
    tags_list = serializers.SerializerMethodField()
    featured_image_url = serializers.SerializerMethodField()
    url = serializers.CharField(source='get_url')
    
    class Meta:
        model = BlogPost
        fields = [
            'id', 'title', 'slug', 'excerpt', 'date', 'author', 
            'categories', 'tags_list', 'featured_image_url',
            'estimated_reading_time', 'view_count', 'url'
        ]
    
    def get_tags_list(self, obj):
        """Get list of tag names."""
        return [tag.name for tag in obj.tags.all()]
    
    def get_featured_image_url(self, obj):
        """Get featured image URL."""
        if obj.featured_image:
            return obj.featured_image.file.url
        return None


class BlogPostDetailSerializer(BlogPostListSerializer):
    """Serializer for blog post detail view."""
    
    body_text = serializers.SerializerMethodField()
    related_posts = serializers.SerializerMethodField()
    
    class Meta(BlogPostListSerializer.Meta):
        fields = BlogPostListSerializer.Meta.fields + [
            'body', 'body_text', 'related_posts'
        ]
    
    def get_body_text(self, obj):
        """Get plain text version of body."""
        return str(obj.body)
    
    def get_related_posts(self, obj):
        """Get related posts."""
        related = BlogPost.objects.live().public().exclude(
            id=obj.id
        ).filter(
            categories__in=obj.categories.all()
        ).distinct()[:3]
        
        return BlogPostListSerializer(
            related, 
            many=True, 
            context=self.context
        ).data
