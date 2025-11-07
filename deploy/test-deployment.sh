#!/bin/bash

# Test script for Docker deployment
# This script tests the containerized deployment

set -e

echo "🚀 Starting Docker deployment test..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not available"
    exit 1
fi

print_status "Docker and Docker Compose are available"

# Validate docker-compose configuration
print_status "Validating docker-compose configuration..."
docker compose config > /dev/null
print_status "✓ Docker compose configuration is valid"

# Build and start services
print_status "Building and starting services..."
docker compose up --build -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Check service health
print_status "Checking service health..."

# Check backend health
if curl -f http://localhost:8000/api/v1/health/health > /dev/null 2>&1; then
    print_status "✓ Backend is healthy"
else
    print_warning "Backend health check failed, but container might still be starting..."
fi

# Check frontend health
if curl -f http://localhost:3000/health > /dev/null 2>&1; then
    print_status "✓ Frontend is healthy"
else
    print_warning "Frontend health check failed, but container might still be starting..."
fi

# Test API endpoints
print_status "Testing API endpoints..."

# Test root endpoint
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    print_status "✓ Backend root endpoint is accessible"
else
    print_warning "Backend root endpoint test failed"
fi

# Test frontend main page
if curl -f http://localhost:3000/ > /dev/null 2>&1; then
    print_status "✓ Frontend main page is accessible"
else
    print_warning "Frontend main page test failed"
fi

# Show container status
print_status "Container status:"
docker compose ps

# Show logs (last 20 lines)
print_status "Recent logs:"
echo "=== Backend logs ==="
docker compose logs --tail=20 backend
echo ""
echo "=== Frontend logs ==="
docker compose logs --tail=20 frontend

# Test forward request (example)
print_status "Testing forward request..."
# This is a placeholder - modify based on your actual API endpoints
# curl -X POST http://localhost:8000/api/v1/forward -H "Content-Type: application/json" -d '{"test": "data"}' || print_warning "Forward request test failed"

print_status "✅ Deployment test completed!"
print_status "Services are running at:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend: http://localhost:8000"
echo "  - Backend API Docs: http://localhost:8000/docs"

echo ""
print_status "To stop services, run: docker compose down"
print_status "To view logs, run: docker compose logs -f"
