# Site Structure

## Overview

The MKT project follows a hierarchical site structure built on Wagtail's page tree system, providing flexible content management and SEO-friendly URLs.

## Page Hierarchy

```bash
Home Page (/)
├── Blog Index (/blog/)
│   ├── Blog Post (/blog/post-slug/)
│   ├── Category Pages (/blog/category/category-slug/)
│   └── Tag Pages (/blog/tag/tag-name/)
├── About (/about/)
├── Contact (/contact/)
└── Search (/search/)

Admin Areas:
├── Wagtail Admin (/admin/)
├── Django Admin (/django-admin/)
└── Monitoring Dashboard (/monitoring/dashboard/)

API Endpoints:
├── Blog API (/blog/api/)
│   ├── Posts (/blog/api/posts/)
│   ├── Categories (/blog/api/categories/)
│   ├── Authors (/blog/api/authors/)
│   └── Stats (/blog/api/stats/)
├── Monitoring API (/monitoring/api/)
└── Wagtail API (/api/v2/)
```

## Page Types

### 1. Home Page

- **Template**: `home/home_page.html`
- **Purpose**: Landing page with site introduction
- **Features**:
  - Hero section
  - Recent blog posts
  - Site navigation
  - SEO optimization

### 2. Blog Index Page

- **Template**: `blog/blog_index_page.html`
- **Purpose**: Main blog listing page
- **Features**:
  - Paginated post listings
  - Category filtering
  - Tag filtering
  - Search functionality
  - RSS feed

### 3. Blog Post Page

- **Template**: `blog/blog_post.html`
- **Purpose**: Individual blog post display
- **Features**:
  - Rich content with StreamField
  - Author information
  - Category and tag display
  - Related posts
  - Social sharing
  - Reading time estimation
  - View count tracking

### 4. Standard Page

- **Template**: `home/standard_page.html`
- **Purpose**: Generic content pages (About, Contact, etc.)
- **Features**:
  - Rich text content
  - Image support
  - SEO fields
  - Breadcrumb navigation

## URL Structure

### Blog URLs

```bash
/blog/                          # Blog index
/blog/page/2/                   # Paginated blog index
/blog/category/technology/      # Category filtered posts
/blog/tag/django/              # Tag filtered posts
/blog/my-first-post/           # Individual blog post
```

### API URLs

```bash
/blog/api/posts/               # Blog posts API
/blog/api/posts/?category=tech # Filtered posts
/blog/api/posts/my-post/       # Single post API
/blog/api/categories/          # Categories API
/blog/api/authors/             # Authors API
/blog/api/stats/               # Blog statistics
```

### Admin URLs

```bash
/admin/                        # Wagtail admin
/admin/pages/                  # Page management
/admin/images/                 # Image library
/admin/documents/              # Document library
/admin/snippets/               # Snippet management
/monitoring/dashboard/         # Monitoring dashboard
```

## Navigation Structure

### Primary Navigation

```bash
┌─────────────────────────────────────────────────────────┐
│  Logo    Home    Blog    About    Contact    Search     │
└─────────────────────────────────────────────────────────┘
```

### Blog Navigation

```bash
┌─────────────────────────────────────────────────────────┐
│  All Posts  │  Categories  │  Tags  │  Authors  │  RSS  │
└─────────────────────────────────────────────────────────┘
```

### Footer Navigation

```bash
┌─────────────────────────────────────────────────────────┐
│  Quick Links    │    Categories    │    Social Media    │
│  - Home         │    - Technology  │    - Twitter       │
│  - Blog         │    - Design      │    - LinkedIn      │
│  - About        │    - Business    │    - GitHub        │
│  - Contact      │    - Tutorials   │    - RSS Feed     │
└─────────────────────────────────────────────────────────┘
```

## Content Organization

### Blog Content Structure

```bash
Blog Post
├── Title
├── Slug (auto-generated)
├── Excerpt (SEO description)
├── Featured Image
├── Body Content (StreamField)
│   ├── Headings
│   ├── Paragraphs
│   ├── Images
│   ├── Code blocks
│   ├── Quotes
│   └── HTML blocks
├── Author
├── Categories (many-to-many)
├── Tags (many-to-many)
├── Publication Date
├── Reading Time (auto-calculated)
└── View Count (auto-tracked)
```

### Category Structure

```bash
Categories
├── Technology
│   ├── Web Development
│   ├── Mobile Development
│   └── DevOps
├── Design
│   ├── UI/UX
│   ├── Graphic Design
│   └── Web Design
├── Business
│   ├── Marketing
│   ├── Strategy
│   └── Analytics
└── Tutorials
    ├── Beginner
    ├── Intermediate
    └── Advanced
```

## SEO Structure

### URL Patterns

- Clean, descriptive URLs
- Hierarchical structure
- Keyword-rich slugs
- No unnecessary parameters

### Meta Information

```html
<!-- Page Title -->
<title>Post Title | Blog | MKT</title>

<!-- Meta Description -->
<meta name="description" content="Post excerpt for SEO">

<!-- Open Graph -->
<meta property="og:title" content="Post Title">
<meta property="og:description" content="Post excerpt">
<meta property="og:image" content="featured-image.jpg">
<meta property="og:type" content="article">

<!-- Twitter Cards -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Post Title">
<meta name="twitter:description" content="Post excerpt">

<!-- Structured Data -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "Post Title",
  "description": "Post excerpt",
  "author": {"@type": "Person", "name": "Author Name"}
}
</script>
```

## Responsive Design Structure

### Breakpoints

```css
/* Mobile First Approach */
/* Base styles: 320px+ */

/* Small tablets: 768px+ */
@media (min-width: 768px) { }

/* Large tablets/small desktops: 1024px+ */
@media (min-width: 1024px) { }

/* Large desktops: 1200px+ */
@media (min-width: 1200px) { }
```

### Layout Structure

```bash
Desktop Layout:
┌─────────────────────────────────────────────────────────┐
│                    Header Navigation                    │
├─────────────────────────────────────────────────────────┤
│  Sidebar (25%)  │         Main Content (75%)           │
│  - Categories   │         - Blog Posts                  │
│  - Recent Posts │         - Pagination                  │
│  - Tags         │         - Related Content             │
└─────────────────────────────────────────────────────────┘

Mobile Layout:
┌─────────────────────────────────────────────────────────┐
│                    Header Navigation                    │
├─────────────────────────────────────────────────────────┤
│                   Main Content (100%)                   │
│                   - Blog Posts                          │
│                   - Pagination                          │
├─────────────────────────────────────────────────────────┤
│                   Sidebar (100%)                        │
│                   - Categories                          │
│                   - Tags                                │
└─────────────────────────────────────────────────────────┘
```

## Search Structure

### Search Functionality

- Full-text search across blog posts
- Category and tag filtering
- Author filtering
- Date range filtering
- Search result highlighting

### Search Index

```bash
Searchable Fields:
├── Blog Posts
│   ├── Title (high weight)
│   ├── Excerpt (medium weight)
│   ├── Body content (low weight)
│   └── Tags (medium weight)
├── Categories
│   ├── Name
│   └── Description
└── Authors
    ├── Name
    └── Bio
```

## Sitemap Structure

### XML Sitemap

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://example.com/blog/</loc>
    <changefreq>daily</changefreq>
    <priority>0.9</priority>
  </url>
  <!-- Blog posts with lastmod dates -->
  <!-- Category pages -->
  <!-- Static pages -->
</urlset>
```

### RSS Feed Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>MKT Blog</title>
    <description>Latest blog posts from MKT</description>
    <link>https://example.com/blog/</link>
    <item>
      <title>Blog Post Title</title>
      <description>Post excerpt</description>
      <link>https://example.com/blog/post-slug/</link>
      <pubDate>Date</pubDate>
      <guid>https://example.com/blog/post-slug/</guid>
    </item>
  </channel>
</rss>
```

## Performance Considerations

### Page Loading Strategy

- Critical CSS inlined
- Non-critical CSS loaded asynchronously
- JavaScript loaded with defer/async
- Images lazy-loaded
- Font loading optimization

### Caching Strategy

- Page-level caching for static content
- Fragment caching for dynamic components
- Browser caching for static assets
- CDN caching for global distribution

## Accessibility Structure

### Semantic HTML

- Proper heading hierarchy (h1-h6)
- Semantic elements (article, section, nav)
- ARIA labels where needed
- Alt text for images
- Focus management

### Navigation Accessibility

- Skip links for keyboard navigation
- Breadcrumb navigation
- Clear link text
- Keyboard-accessible menus
- Screen reader friendly
