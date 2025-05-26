# blog/models.py
from django.db import models
from django.core.paginator import Paginator
from django.shortcuts import render
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse
from django.http import Http404

from wagtail.models import Page, Orderable
from wagtail.fields import RichTextField, StreamField
from wagtail.admin.panels import FieldPanel, InlinePanel
from wagtail.search import index
from wagtail.snippets.models import register_snippet
from wagtail.api import APIField
from wagtail import blocks
from wagtail.images.blocks import ImageChooserBlock
from wagtail.contrib.routable_page.models import RoutablePageMixin, route

from modelcluster.fields import ParentalKey, ParentalManyToManyField
from modelcluster.contrib.taggit import ClusterTaggableManager
from taggit.models import TaggedItemBase
from monitoring.metrics import (
    record_blog_post_created,
    record_blog_post_view,
    BLOG_POST_READING_TIME
)

import logging

logger = logging.getLogger(__name__)


class BlogStreamBlock(blocks.StreamBlock):
    """Custom stream block for blog content."""
    
    heading = blocks.CharBlock(classname="full title")
    paragraph = blocks.RichTextBlock()
    image = ImageChooserBlock()
    quote = blocks.BlockQuoteBlock()
    code = blocks.TextBlock(classname="full code")
    html = blocks.RawHTMLBlock(classname="full")
    
    class Meta:
        icon = 'edit'


@register_snippet
class BlogCategory(models.Model):
    """Blog category snippet."""
    
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True, max_length=100)
    description = models.TextField(blank=True)
    created_date = models.DateTimeField(auto_now_add=True)
    
    panels = [
        FieldPanel('name'),
        FieldPanel('slug'),
        FieldPanel('description'),
    ]
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name_plural = 'Blog Categories'


@register_snippet
class BlogTag(TaggedItemBase):
    """Blog tag model."""
    
    content_object = ParentalKey(
        'BlogPost',
        related_name='tagged_items',
        on_delete=models.CASCADE
    )


class BlogAuthor(models.Model):
    """Blog author model."""
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(max_length=500, blank=True)
    website = models.URLField(blank=True)
    twitter = models.CharField(max_length=100, blank=True)
    linkedin = models.CharField(max_length=100, blank=True)
    created_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username}"
    
    @property
    def name(self):
        return self.user.get_full_name() or self.user.username


class BlogIndexPage(RoutablePageMixin, Page):
    """Main blog index page with routing capabilities."""
    
    intro = RichTextField(blank=True)
    posts_per_page = models.IntegerField(default=10)
    
    content_panels = Page.content_panels + [
        FieldPanel('intro'),
        FieldPanel('posts_per_page'),
    ]
    
    api_fields = [
        APIField('intro'),
        APIField('posts_per_page'),
    ]
    
    def get_context(self, request):
        """Get context for blog index."""
        context = super().get_context(request)
        
        # Get all published blog posts
        blog_posts = BlogPost.objects.live().public().order_by('-first_published_at')
        
        # Pagination
        paginator = Paginator(blog_posts, self.posts_per_page)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        context.update({
            'blog_posts': page_obj,
            'categories': BlogCategory.objects.all(),
        })
        
        logger.info(f"Blog index loaded {len(page_obj)} posts")
        return context

    # Fix the route decorators (lines around 120-140)

    @route(r'^category/(?P<category_slug>[-\w]+)/$')  # Fixed: removed extra backticks
    def category_view(self, request, category_slug):
        """Category filtered view."""
        try:
            category = BlogCategory.objects.get(slug=category_slug)
            blog_posts = BlogPost.objects.live().public().filter(
                categories=category
            ).order_by('-first_published_at')
            
            paginator = Paginator(blog_posts, self.posts_per_page)
            page_number = request.GET.get('page')
            page_obj = paginator.get_page(page_number)
            
            context = self.get_context(request)
            context.update({
                'blog_posts': page_obj,
                'current_category': category,
                'page_title': f'Category: {category.name}',
            })
            
            logger.info(f"Category {category.name} loaded {len(page_obj)} posts")
            return render(request, 'blog/blog_index_page.html', context)
            
        except BlogCategory.DoesNotExist:
            logger.warning(f"Category not found: {category_slug}")
            raise Http404("Category not found")
    
    @route(r'^tag/(?P<tag>[-\w]+)/$')  # Fixed: removed extra backticks
    def tag_view(self, request, tag):
        """Tag filtered view."""
        blog_posts = BlogPost.objects.live().public().filter(
            tags__name=tag
        ).order_by('-first_published_at')
        
        paginator = Paginator(blog_posts, self.posts_per_page)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        context = self.get_context(request)
        context.update({
            'blog_posts': page_obj,
            'current_tag': tag,
            'page_title': f'Tag: {tag}',
        })
        
        logger.info(f"Tag {tag} loaded {len(page_obj)} posts")
        return render(request, 'blog/blog_index_page.html', context)


class BlogPost(Page):
    """Individual blog post model."""
    
    # Content fields
    excerpt = models.TextField(
        max_length=300,
        help_text="Brief description of the post for listings and SEO"
    )
    body = StreamField(
        BlogStreamBlock(),
        use_json_field=True,
        blank=True
    )
    
    # Meta fields
    date = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(
        BlogAuthor, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    categories = ParentalManyToManyField('BlogCategory', blank=True)
    tags = ClusterTaggableManager(through=BlogTag, blank=True)
    
    # SEO fields
    featured_image = models.ForeignKey(
        'wagtailimages.Image',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='+',
        help_text="Featured image for social sharing and listings"
    )
    
    # Reading time estimation
    estimated_reading_time = models.PositiveIntegerField(
        default=5,
        help_text="Estimated reading time in minutes"
    )
    
    # Analytics
    view_count = models.PositiveIntegerField(default=0)
    
    # Search fields
    search_fields = Page.search_fields + [
        index.SearchField('excerpt'),
        index.SearchField('body'),
        index.FilterField('date'),
        index.FilterField('author'),
        index.RelatedFields('categories', [
            index.SearchField('name'),
        ]),
        index.RelatedFields('tags', [
            index.SearchField('name'),
        ]),
    ]
    
    # Admin panels
    content_panels = Page.content_panels + [
        FieldPanel('excerpt'),
        FieldPanel('body'),
        FieldPanel('featured_image'),
        FieldPanel('date'),
        FieldPanel('author'),
        FieldPanel('categories'),
        FieldPanel('tags'),
        FieldPanel('estimated_reading_time'),
    ]
    
    # API fields
    api_fields = [
        APIField('excerpt'),
        APIField('body'),
        APIField('date'),
        APIField('author'),
        APIField('categories'),
        APIField('tags'),
        APIField('featured_image'),
        APIField('estimated_reading_time'),
        APIField('view_count'),
    ]
    
    parent_page_types = ['BlogIndexPage']
    
    def get_context(self, request):
        """Get context for blog post."""
        context = super().get_context(request)
        
        # Increment view count
        self.view_count += 1
        self.save(update_fields=['view_count'])
        
        # Record blog post view metric
        category_name = self.categories.first().name if self.categories.exists() else 'Uncategorized'
        record_blog_post_view(self.slug, category_name)
        
        # Get related posts - ensure we're getting actual posts
        related_posts = BlogPost.objects.live().public().exclude(
            id=self.id
        ).filter(
            categories__in=self.categories.all()
        ).distinct().order_by('-first_published_at')[:3]
        
        # Make sure we have related posts
        if not related_posts:
            # If no related posts by category, get latest posts
            related_posts = BlogPost.objects.live().public().exclude(
                id=self.id
            ).order_by('-first_published_at')[:3]
        
        context.update({
            'related_posts': related_posts,
        })
        
        logger.info(f"Blog post {self.title} viewed (total: {self.view_count})")
        return context

    
    def save(self, *args, **kwargs):
        """Override save to calculate reading time and record metrics."""
        is_new = self.pk is None
        if hasattr(self, 'body') and self.body:
            # Extract text content from StreamField for more accurate word count
            content = ""
            for block in self.body:
                if block.block_type == 'paragraph':
                    content += str(block.value)
                elif block.block_type == 'heading':
                    content += str(block.value) + " "
                elif block.block_type == 'quote':
                    content += str(block.value) + " "
                elif block.block_type == 'code':
                    content += str(block.value) + " "
            # Count words and calculate reading time
            word_count = len(content.split())
            self.estimated_reading_time = max(1, word_count // 200)  # 200 words per minute
        
        if self.body:
            # Simple word count estimation
            word_count = len(str(self.body).split())
            self.estimated_reading_time = max(1, word_count // 200)  # 200 words per minute
            
            # Record reading time metric
            if self.categories.exists():
                category_name = self.categories.first().name
                BLOG_POST_READING_TIME.labels(category=category_name).observe(
                    self.estimated_reading_time * 60  # Convert to seconds
                )
                
            # Count words and calculate reading time
            word_count = len(content.split())
            self.estimated_reading_time = max(1, word_count // 200)  # 200 words per minute
        
        super().save(*args, **kwargs)
         # Record blog post creation metric
        if is_new and self.live:
            author_name = self.author.name if self.author else 'Unknown'
            category_name = self.categories.first().name if self.categories.exists() else 'Uncategorized'
            record_blog_post_created(author_name, category_name)


    
    def get_absolute_url(self):
        """Get absolute URL for the post."""
        return reverse('blog:post_detail', args=[self.slug])
    
    @property
    def blog_index(self):
        """Get the blog index page."""
        return self.get_parent().specific
    
    class Meta:
        ordering = ['-first_published_at']