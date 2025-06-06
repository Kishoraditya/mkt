from django.conf import settings
from django.urls import include, path
from django.contrib import admin

from wagtail.admin import urls as wagtailadmin_urls
from wagtail import urls as wagtail_urls
from wagtail.documents import urls as wagtaildocs_urls

from search import views as search_views

from django.conf.urls.static import static
from django.views.generic import TemplateView
from wagtail.contrib.sitemaps.views import sitemap

urlpatterns = [
    path("django-admin/", admin.site.urls),
    path("admin/", include(wagtailadmin_urls)),
    path("documents/", include(wagtaildocs_urls)),
    path('search/', include('search.urls')),
    path('blog/', include(('blog.urls', 'blog'), namespace='blog')),
    # Monitoring URLs
    path("monitoring/", include("monitoring.urls", namespace="monitoring")),
    path('communication/', include('communication.urls')),
    path('api/v1/blog/', include('blog.api_urls')),
    path('api/v1/monitoring/', include('monitoring.api_urls')),
    path('api/v1/communication/', include('communication.api_urls')), 
    # Prometheus metrics
    path('', include('django_prometheus.urls')),
    
    # Sitemap
    path('sitemap.xml', sitemap, name='sitemap'),
    
    # Documentation
    path('docs/', TemplateView.as_view(
        template_name='docs/index.html'
    ), name='documentation'),
]


if settings.DEBUG:
    from django.conf.urls.static import static
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns

    # Serve static and media files from development server
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns = urlpatterns + [
    # For anything not caught by a more specific rule above, hand over to
    # Wagtail's page serving mechanism. This should be the last pattern in
    # the list:
    path("", include(wagtail_urls)),
    # Alternatively, if you want Wagtail pages to be served from a subpath
    # of your site, rather than the site root:
    #    path("pages/", include(wagtail_urls)),
]
