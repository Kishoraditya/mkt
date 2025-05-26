# blog/tests.py
import json
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from wagtail.models import Site, Page
from wagtail.test.utils import WagtailPageTests
from wagtail.images.tests.utils import Image, get_test_image_file
from rest_framework.test import APITestCase
from rest_framework import status
from django.utils import timezone
from home.models import HomePage
from .models import BlogIndexPage, BlogPost, BlogCategory, BlogAuthor


class BlogModelTests(TestCase):
    """Test blog models functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create site and pages
        self.site = Site.objects.get(is_default_site=True)
        self.home_page = HomePage.objects.get(slug='home')
        
        # Create blog index page
        self.blog_index = BlogIndexPage(
            title="Blog",
            slug="blog",
            intro="Welcome to our blog"
        )
        self.home_page.add_child(instance=self.blog_index)
        
        # Create user and author
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
        self.author = BlogAuthor.objects.create(
            user=self.user,
            bio="Test author bio",
            website="https://example.com",
            twitter="testuser"
        )
        
        # Create category
        self.category = BlogCategory.objects.create(
            name="Technology",
            slug="technology",
            description="Tech-related posts"
        )
        
        # Create test image
        self.image = Image.objects.create(
            title="Test image",
            file=get_test_image_file()
        )
    
    def test_blog_category_str(self):
        """Test BlogCategory string representation."""
        self.assertEqual(str(self.category), "Technology")
    
    def test_blog_author_str(self):
        """Test BlogAuthor string representation."""
        self.assertEqual(str(self.author), "Test User")
    
    def test_blog_author_name_property(self):
        """Test BlogAuthor name property."""
        self.assertEqual(self.author.name, "Test User")
        
        # Test with user without full name
        user_no_name = User.objects.create_user(
            username='noname',
            email='noname@example.com'
        )
        author_no_name = BlogAuthor.objects.create(user=user_no_name)
        self.assertEqual(author_no_name.name, "noname")
    
    def test_blog_index_page_context(self):
        """Test BlogIndexPage context method."""
        # Create some blog posts
        for i in range(15):
            post = BlogPost(
                title=f"Test Post {i}",
                slug=f"test-post-{i}",
                excerpt=f"Excerpt for post {i}",
                author=self.author
            )
            self.blog_index.add_child(instance=post)
        
        # Test context
        request = self.client.get(self.blog_index.url).wsgi_request
        context = self.blog_index.get_context(request)
        
        self.assertIn('blog_posts', context)
        self.assertIn('categories', context)
        self.assertEqual(len(context['blog_posts']), 10)  # Default posts_per_page
    
    def test_blog_post_save_reading_time(self):
        """Test BlogPost reading time calculation."""
        # Create post with content
        body_content = [
            ('paragraph', ' '.join(["word"] * 400))  # 400 words
        ]
        post = BlogPost(
            title="Test Post",
            slug="test-post",
            excerpt="Test excerpt",
            body=body_content,
            author=self.author
        )
        self.blog_index.add_child(instance=post)
        post.save()  # Explicitly call save to trigger reading time calculation
        
        # Should be 2 minutes (400 words / 200 wpm)
        self.assertEqual(post.estimated_reading_time, 2)

    
    def test_blog_post_get_context(self):
        """Test BlogPost context method."""
        # Create main post
        post = BlogPost(
            title="Main Post",
            slug="main-post",
            excerpt="Main post excerpt",
            author=self.author
        )
        self.blog_index.add_child(instance=post)
        post.categories.add(self.category)
        
        # Create related posts
        for i in range(5):
            related_post = BlogPost(
                title=f"Related Post {i}",
                slug=f"related-post-{i}",
                excerpt=f"Related excerpt {i}",
                author=self.author
            )
            self.blog_index.add_child(instance=related_post)
            related_post.categories.add(self.category)
        
        # Test context
        request = self.client.get(post.url).wsgi_request
        context = post.get_context(request)
        
        self.assertIn('related_posts', context)
        self.assertEqual(len(context['related_posts']), 3)  # Limited to 3
        
        # Check view count increment
        post.refresh_from_db()
        self.assertEqual(post.view_count, 1)


class BlogPageTests(WagtailPageTests):
    """Test blog page functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.home_page = HomePage.objects.get(slug='home')
        self.blog_index = BlogIndexPage(
            title="Blog",
            slug="blog"
        )
        self.home_page.add_child(instance=self.blog_index)
        
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.author = BlogAuthor.objects.create(user=self.user)
        
        # Create category
        self.category = BlogCategory.objects.create(
            name="Tech",
            slug="tech"
        )
    
    def test_can_create_blog_index_page(self):
        """Test creating BlogIndexPage."""
        self.assertCanCreateAt(HomePage, BlogIndexPage)
    
    def test_can_create_blog_post(self):
        """Test creating BlogPost."""
        self.assertCanCreateAt(BlogIndexPage, BlogPost)
        self.assertCanNotCreateAt(HomePage, BlogPost)
    
    def test_blog_index_routing(self):
        """Test BlogIndexPage routing functionality."""
        # Create category with a unique slug
        import uuid
        unique_slug = f"tech-{uuid.uuid4().hex[:6]}"
        
        category = BlogCategory.objects.create(
            name="Tech",
            slug=unique_slug
        )
        
        # Create post with category
        post = BlogPost(
            title="Tech Post",
            slug=f"tech-post-{uuid.uuid4().hex[:6]}",
            excerpt="Tech excerpt",
            author=self.author
        )
        self.blog_index.add_child(instance=post)
        
        # Publish the post
        revision = post.save_revision()
        revision.publish()
        
        # Add category to the post
        post.categories.add(category)
        post.save()
        
        # Test category route
        response = self.client.get(f"{self.blog_index.url}category/{unique_slug}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Tech Post")
        
        # Test tag route (add tag first)
        tag_name = "django"
        post.tags.add(tag_name)
        post.save()
        
        # Test tag route
        response = self.client.get(f"{self.blog_index.url}tag/{tag_name}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Tech Post")



# Add these imports at the top if not already present
from django.test import override_settings
from wagtail.test.utils import WagtailTestUtils
from django.urls import reverse

class BlogAPITests(APITestCase):
    """Test blog API endpoints."""
    
    def setUp(self):
        """Set up test data."""
        # Create pages
        self.home_page = HomePage.objects.get(slug='home')
        self.blog_index = BlogIndexPage(
            title="Blog",
            slug="blog"
        )
        self.home_page.add_child(instance=self.blog_index)
        
        # Create user and author
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            first_name='Test',
            last_name='Author'
        )
        self.author = BlogAuthor.objects.create(user=self.user)
        
        # Create category
        self.category = BlogCategory.objects.create(
            name="Technology",
            slug="technology"
        )
        
        # Create test posts
        for i in range(5):
            post = BlogPost(
                title=f"Test Post {i}",
                slug=f"test-post-{i}",
                excerpt=f"Excerpt for post {i}",
                author=self.author,
                view_count=i * 10
            )
            self.blog_index.add_child(instance=post)
            post.categories.add(self.category)
            post.tags.add(f"tag{i}")
            # Make sure the post is published
            post.live = True
            post.save()
    
    def test_blog_category_list_api(self):
        """Test blog category list API."""
        url = reverse('blog:category_list_api')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()['results']  # Access the 'results' key for paginated data
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], 'Technology')
    
    def test_blog_author_list_api(self):
        """Test blog author list API."""
        url = reverse('blog:author_list_api')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()['results']  # Access the 'results' key for paginated data
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], 'Test Author')
    
    @override_settings(REST_FRAMEWORK={
        'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
        'PAGE_SIZE': 20,
    })
    def test_blog_post_list_pagination(self):
        """Test blog post list API pagination."""
        # Create 25 posts for pagination testing
        for i in range(5, 25):
            post = BlogPost(
                title=f"Pagination Post {i}",
                slug=f"pagination-post-{i}",
                excerpt=f"Pagination excerpt {i}",
                author=self.author
            )
            self.blog_index.add_child(instance=post)
            post.categories.add(self.category)
            post.live = True
            post.save()
        
        url = reverse('blog:post_list_api')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['results']), 20)
        self.assertTrue(data['next'])  # Should have a next page
        
        # Test second page
        response = self.client.get(data['next'])
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['results']), 5)  # 5 remaining posts
        self.assertIsNone(data['next'])  # No more pages
    
    def test_blog_post_list_filtering(self):
        """Test blog post list API filtering."""
        url = reverse('blog:post_list_api')
        
        # Test category filtering
        response = self.client.get(f"{url}?category=technology")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['results']), 5)
        
        # Test tag filtering
        response = self.client.get(f"{url}?tag=tag0")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['results']), 1)
        self.assertEqual(data['results'][0]['title'], "Test Post 0")
        
        # Test search filtering
        response = self.client.get(f"{url}?search=Post 3")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['results']), 1)
        self.assertEqual(data['results'][0]['title'], "Test Post 3")
    
    def test_combined_filtering_and_ordering(self):
        """Test combining filtering and ordering in API."""
        # Create posts with a different category
        programming_category = BlogCategory.objects.create(
            name="Programming",
            slug="programming"
        )
        
        for i in range(3):
            post = BlogPost(
                title=f"Programming Post {i}",
                slug=f"programming-post-{i}",
                excerpt=f"Programming excerpt {i}",
                author=self.author,
                view_count=i * 5
            )
            self.blog_index.add_child(instance=post)
            post.categories.add(programming_category)
            post.live = True
            post.save()
        
        url = reverse('blog:post_list_api')
        
        # Test filtering by category and ordering by view_count
        response = self.client.get(f"{url}?category=programming&ordering=-view_count")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['results']), 3)
        self.assertEqual(data['results'][0]['title'], "Programming Post 2")  # Highest view count


# Add this class to blog/tests.py
# Add this new test class to blog/tests.py

class BlogTemplateTests(TestCase):
    """Test blog template rendering."""
    
    def setUp(self):
        """Set up test data."""
        # Create site and pages
        self.site = Site.objects.get(is_default_site=True)
        self.home_page = HomePage.objects.get(slug='home')
        
        # Create blog index page
        self.blog_index = BlogIndexPage(
            title="Blog",
            slug="blog",
            intro="Welcome to our blog"
        )
        self.home_page.add_child(instance=self.blog_index)
        
        # Create user and author
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
        self.author = BlogAuthor.objects.create(
            user=self.user,
            bio="Test author bio"
        )
        
        # Create category
        self.category = BlogCategory.objects.create(
            name="Technology",
            slug="technology"
        )
        
        # Create a blog post with the specific title we're testing for
        self.blog_post = BlogPost(
            title="Test Blog Post",
            slug="test-blog-post",
            excerpt="Test blog post excerpt",
            author=self.author
        )
        self.blog_index.add_child(instance=self.blog_post)
        self.blog_post.categories.add(self.category)
        
        # Publish the post to make it visible in the index
        revision = self.blog_post.save_revision()
        revision.publish()
    
    def test_blog_index_template(self):
        """Test blog index template."""
        response = self.client.get(self.blog_index.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Blog Post")
    
    def test_blog_post_template(self):
        """Test blog post template."""
        response = self.client.get(self.blog_post.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Blog Post")
        self.assertContains(response, "Test blog post excerpt")
    
    def test_blog_category_template(self):
        """Test blog category template."""
        response = self.client.get(f"{self.blog_index.url}category/technology/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Blog Post")
        self.assertContains(response, "Technology")
    
    def test_blog_tag_template(self):
        """Test blog tag template."""
        # Add a tag to the post
        self.blog_post.tags.add("wagtail")
        response = self.client.get(f"{self.blog_index.url}tag/wagtail/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Blog Post")
        self.assertContains(response, "wagtail")
