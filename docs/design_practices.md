# Design Practices

## Overview

The MKT project follows modern web design principles, accessibility standards, and performance best practices to deliver an exceptional user experience across all devices and platforms.

## Design Philosophy

### Core Principles

1. **User-Centered Design**: Every design decision prioritizes user needs and experience
2. **Accessibility First**: Inclusive design for all users, including those with disabilities
3. **Performance Focused**: Fast loading times and smooth interactions
4. **Mobile First**: Responsive design starting from mobile devices
5. **Content First**: Design serves content, not the other way around
6. **Consistency**: Unified visual language across all pages and components

### Design Goals

- **Clarity**: Clear information hierarchy and intuitive navigation
- **Simplicity**: Clean, uncluttered interfaces that focus on content
- **Efficiency**: Quick task completion and easy content discovery
- **Engagement**: Visually appealing design that encourages interaction
- **Trust**: Professional appearance that builds credibility

## Visual Design System

### Color Palette

```css
:root {
  /* Primary Colors */
  --primary-50: #eff6ff;
  --primary-100: #dbeafe;
  --primary-500: #3b82f6;
  --primary-600: #2563eb;
  --primary-700: #1d4ed8;
  --primary-900: #1e3a8a;

  /* Neutral Colors */
  --gray-50: #f9fafb;
  --gray-100: #f3f4f6;
  --gray-200: #e5e7eb;
  --gray-300: #d1d5db;
  --gray-400: #9ca3af;
  --gray-500: #6b7280;
```css
  border: 2px solid var(--gray-300);
  border-top-color: var(--primary-500);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Page transitions */
.page-enter {
  opacity: 0;
  transform: translateY(20px);
}

.page-enter-active {
  opacity: 1;
  transform: translateY(0);
  transition: opacity 0.3s ease, transform 0.3s ease;
}
```

### Reduced Motion Support

```css
/* Respect user preferences */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Alternative for reduced motion */
@media (prefers-reduced-motion: reduce) {
  .animate-bounce {
    animation: none;
  }
  
  .animate-pulse {
    animation: none;
    opacity: 1;
  }
}
```

## Dark Mode Support

### CSS Custom Properties Approach

```css
/* Light theme (default) */
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f9fafb;
  --text-primary: #111827;
  --text-secondary: #6b7280;
  --border-color: #e5e7eb;
}

/* Dark theme */
@media (prefers-color-scheme: dark) {
  :root {
    --bg-primary: #111827;
    --bg-secondary: #1f2937;
    --text-primary: #f9fafb;
    --text-secondary: #d1d5db;
    --border-color: #374151;
  }
}

/* Manual dark mode toggle */
[data-theme="dark"] {
  --bg-primary: #111827;
  --bg-secondary: #1f2937;
  --text-primary: #f9fafb;
  --text-secondary: #d1d5db;
  --border-color: #374151;
}
```

### Dark Mode JavaScript

```javascript
// Dark mode toggle
function toggleDarkMode() {
  const html = document.documentElement;
  const currentTheme = html.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  
  html.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
}

// Initialize theme
function initializeTheme() {
  const savedTheme = localStorage.getItem('theme');
  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = savedTheme || (systemPrefersDark ? 'dark' : 'light');
  
  document.documentElement.setAttribute('data-theme', theme);
}

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
  if (!localStorage.getItem('theme')) {
    document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
  }
});
```

## Component Design Patterns

### Blog Post Card

```html
<article class="blog-card">
  <div class="blog-card__image">
    <img src="featured-image.jpg" alt="Post title" loading="lazy">
    <div class="blog-card__category">Technology</div>
  </div>
  <div class="blog-card__content">
    <header class="blog-card__header">
      <h3 class="blog-card__title">
        <a href="/blog/post-slug/">Post Title</a>
      </h3>
      <div class="blog-card__meta">
        <time datetime="2024-01-15">Jan 15, 2024</time>
        <span class="blog-card__reading-time">5 min read</span>
      </div>
    </header>
    <p class="blog-card__excerpt">Post excerpt...</p>
    <footer class="blog-card__footer">
      <div class="blog-card__author">
        <img src="author-avatar.jpg" alt="Author name" class="blog-card__avatar">
        <span class="blog-card__author-name">Author Name</span>
      </div>
      <div class="blog-card__tags">
        <span class="tag">Django</span>
        <span class="tag">Python</span>
      </div>
    </footer>
  </div>
</article>
```

```css
.blog-card {
  background: var(--bg-primary);
  border-radius: 0.75rem;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.blog-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
}

.blog-card__image {
  position: relative;
  aspect-ratio: 16 / 9;
  overflow: hidden;
}

.blog-card__image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.blog-card:hover .blog-card__image img {
  transform: scale(1.05);
}

.blog-card__category {
  position: absolute;
  top: 1rem;
  left: 1rem;
  background: var(--primary-600);
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.875rem;
  font-weight: 500;
}

.blog-card__content {
  padding: 1.5rem;
}

.blog-card__title a {
  color: var(--text-primary);
  text-decoration: none;
  font-weight: 600;
  line-height: 1.4;
}

.blog-card__title a:hover {
  color: var(--primary-600);
}

.blog-card__meta {
  display: flex;
  gap: 1rem;
  margin-top: 0.5rem;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.blog-card__excerpt {
  margin: 1rem 0;
  color: var(--text-secondary);
  line-height: 1.6;
}

.blog-card__footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1rem;
}

.blog-card__author {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.blog-card__avatar {
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  object-fit: cover;
}

.blog-card__tags {
  display: flex;
  gap: 0.5rem;
}

.tag {
  background: var(--bg-secondary);
  color: var(--text-secondary);
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 500;
}
```

### Navigation Component

```html
<nav class="main-nav" role="navigation" aria-label="Main navigation">
  <div class="main-nav__container">
    <div class="main-nav__brand">
      <a href="/" class="main-nav__logo">
        <img src="logo.svg" alt="MKT" width="120" height="40">
      </a>
    </div>
    
    <button class="main-nav__toggle" aria-expanded="false" aria-controls="main-menu">
      <span class="sr-only">Toggle navigation</span>
      <span class="hamburger"></span>
    </button>
    
    <div class="main-nav__menu" id="main-menu">
      <ul class="main-nav__list">
        <li class="main-nav__item">
          <a href="/" class="main-nav__link" aria-current="page">Home</a>
        </li>
        <li class="main-nav__item">
          <a href="/blog/" class="main-nav__link">Blog</a>
        </li>
        <li class="main-nav__item">
          <a href="/about/" class="main-nav__link">About</a>
        </li>
        <li class="main-nav__item">
          <a href="/contact/" class="main-nav__link">Contact</a>
        </li>
      </ul>
      
      <div class="main-nav__actions">
        <button class="theme-toggle" aria-label="Toggle dark mode">
          <span class="theme-toggle__icon"></span>
        </button>
        <a href="/search/" class="main-nav__search" aria-label="Search">
          <svg class="search-icon" width="20" height="20">
            <use href="#search-icon"></use>
          </svg>
        </a>
      </div>
    </div>
  </div>
</nav>
```

## Design Testing and Quality Assurance

### Visual Regression Testing

```javascript
// Example with Playwright
const { test, expect } = require('@playwright/test');

test('blog card component visual test', async ({ page }) => {
  await page.goto('/blog/');
  await page.waitForLoadState('networkidle');
  
  const blogCard = page.locator('.blog-card').first();
  await expect(blogCard).toHaveScreenshot('blog-card.png');
});

test('responsive navigation test', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto('/');
  
  const nav = page.locator('.main-nav');
  await expect(nav).toHaveScreenshot('mobile-nav.png');
});
```

### Accessibility Testing

```javascript
// Example with axe-playwright
const { injectAxe, checkA11y } = require('axe-playwright');

test('accessibility test', async ({ page }) => {
  await page.goto('/blog/');
  await injectAxe(page);
  
  await checkA11y(page, null, {
    detailedReport: true,
    detailedReportOptions: { html: true }
  });
});
```

### Performance Testing

```javascript
// Lighthouse CI configuration
module.exports = {
  ci: {
    collect: {
      url: ['http://localhost:8000/', 'http://localhost:8000/blog/'],
      numberOfRuns: 3
    },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['error', { minScore: 0.9 }],
        'categories:seo': ['error', { minScore: 0.9 }]
      }
    }
  }
};
```

## Design Documentation

### Component Documentation

```markdown
## Blog Card Component

### Usage
```html
<article class="blog-card">
  <!-- Component markup -->
</article>
```

### Props

- `title`: Post title (required)
- `excerpt`: Post excerpt (required)
- `image`: Featured image URL (optional)
- `author`: Author information (required)
- `date`: Publication date (required)
- `tags`: Array of tags (optional)

### Variants

- Default: Standard blog card
- Featured: Larger card for featured posts
- Compact: Smaller card for sidebar

### Accessibility

- Semantic HTML structure
- Proper heading hierarchy
- Alt text for images
- Focus management
