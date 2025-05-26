#!/bin/bash

echo "🚀 MKT Project Setup"
echo "===================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check if PostgreSQL is running (optional check)
if ! pg_isready -h localhost -p 5432 &> /dev/null; then
    echo -e "${YELLOW}⚠️ PostgreSQL doesn't seem to be running on localhost:5432${NC}"
    echo -e "${YELLOW}   Make sure your database is configured correctly.${NC}"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️ Please update the .env file with your database credentials${NC}"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p media
mkdir -p static
mkdir -p cache

# Run migrations
echo "🗄️ Running migrations..."
python manage.py makemigrations
python manage.py migrate

# Create superuser prompt
echo ""
read -p "Do you want to create a superuser? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python manage.py createsuperuser
fi

# Collect static files
echo "📁 Collecting static files..."
python manage.py collectstatic --noinput

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "🚀 To start the application:"
echo -e "${GREEN}  source venv/bin/activate${NC}"
echo -e "${GREEN}  python manage.py runserver${NC}"
echo ""
echo "🌐 Access points:"
echo "  • Django App: http://localhost:8000"
echo "  • Admin: http://localhost:8000/admin/"
echo "  • Blog: http://localhost:8000/blog/"
echo "  • Monitoring: http://localhost:8000/monitoring/dashboard/"
echo ""
echo "📊 For monitoring setup:"
echo "  • Start Prometheus: docker run -p 9090:9090 -v ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus"
echo "  • Start Grafana: docker run -p 3000:3000 grafana/grafana"
echo "  • Or use: docker-compose up -d"
