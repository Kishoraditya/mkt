from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from wagtail.models import Site
from home.models import HomePage
from blog.models import BlogIndexPage, BlogPost, BlogCategory, BlogAuthor


class Command(BaseCommand):
    help = 'Create sample blog data for testing'

    def handle(self, *args, **options):
        self.stdout.write('Creating sample data...')
        
        # Get or create home page
        try:
            home_page = HomePage.objects.get(slug='home')
        except HomePage.DoesNotExist:
            site = Site.objects.get(is_default_site=True)
            home_page = HomePage(
                title="Welcome to MKT",
                slug="home",
                intro="A modern Django application with Wagtail CMS"
            )
            site.root_page.add_child(instance=home_page)
            site.root_page = home_page
            site.save()
        
        # Create blog index page
        try:
            blog_index = BlogIndexPage.objects.get(slug='blog')
        except BlogIndexPage.DoesNotExist:
            blog_index = BlogIndexPage(
                title="Blog",
                slug="blog",
                intro="Welcome to our blog where we share insights and updates."
            )
            home_page.add_child(instance=blog_index)
        
        # Create sample user and author
        user, created = User.objects.get_or_create(
            username='sampleauthor',
            defaults={
                'email': 'author@example.com',
                'first_name': 'Sample',
                'last_name': 'Author'
            }
        )
        
        author, created = BlogAuthor.objects.get_or_create(
            user=user,
            defaults={
                'bio': 'A passionate writer and developer sharing insights about technology and development.',
                'website': 'https://example.com',
                'twitter': 'sampleauthor'
            }
        )
        
        # Create sample categories
        categories_data = [
            {'name': 'Technology', 'slug': 'technology', 'description': 'Latest in tech'},
            {'name': 'Development', 'slug': 'development', 'description': 'Software development tips'},
            {'name': 'Tutorial', 'slug': 'tutorial', 'description': 'Step-by-step guides'},
        ]
        
        categories = []
        for cat_data in categories_data:
            category, created = BlogCategory.objects.get_or_create(
                slug=cat_data['slug'],
                defaults=cat_data
            )
            categories.append(category)
        
        # Create sample blog posts
        posts_data = [
            {
                'title': 'Getting Started with Django and Wagtail',
                'slug': 'getting-started-django-wagtail',
                'excerpt': 'Learn how to build modern web applications using Django and Wagtail CMS.',
                'body': 'This is a comprehensive guide to getting started with Django and Wagtail...',
                'categories': [categories[0], categories[1]],
                'tags': ['django', 'wagtail', 'python', 'cms']
            },
            {
                'title': 'Building REST APIs with Django REST Framework',
                'slug': 'building-rest-apis-drf',
                'excerpt': 'A complete tutorial on creating robust REST APIs using Django REST Framework.',
                'body': 'REST APIs are essential for modern web applications...',
                'categories': [categories[1], categories[2]],
                'tags': ['api', 'rest', 'django', 'tutorial']
            },
            {
                'title': 'Monitoring Django Applications with Prometheus',
                'slug': 'monitoring-django-prometheus',
                'excerpt': 'Set up comprehensive monitoring for your Django applications using Prometheus and Grafana.',
                'body': 'Monitoring is crucial for production applications...',
                'categories': [categories[0], categories[2]],
                'tags': ['monitoring', 'prometheus', 'grafana', 'devops']
            },
        ]
        
        for post_data in posts_data:
            try:
                post = BlogPost.objects.get(slug=post_data['slug'])
                self.stdout.write(f'Post "{post_data["title"]}" already exists')
            except BlogPost.DoesNotExist:
                post = BlogPost(
                    title=post_data['title'],
                    slug=post_data['slug'],
                    excerpt=post_data['excerpt'],
                    body=post_data['body'],
                    author=author
                )
                blog_index.add_child(instance=post)
                
                # Add categories
                for category in post_data['categories']:
                    post.categories.add(category)
                
                # Add tags
                for tag in post_data['tags']:
                    post.tags.add(tag)
                
                post.save()
                self.stdout.write(f'Created post: "{post_data["title"]}"')
        
        self.stdout.write(
            self.style.SUCCESS('Successfully created sample data!')
        )
