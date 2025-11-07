#!/bin/bash

# Quick start script for Docker deployment
# This script provides easy commands to start the application

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}  Docker Deployment Quick Start     ${NC}"
    echo -e "${BLUE}=====================================${NC}"
    echo ""
}

print_command() {
    echo -e "${GREEN}$1${NC} - $2"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

print_header

echo "Available commands:"
echo ""

print_command "./test-deployment.sh" "Build and test the complete deployment"
print_command "docker compose up --build -d" "Start services in background"
print_command "docker compose up --build" "Start services with logs"
print_command "docker compose down" "Stop and remove services"
print_command "docker compose logs -f" "Follow logs"
print_command "docker compose ps" "Show service status"
echo ""

print_info "Environment Setup:"
echo "1. Copy .env.example to .env and configure as needed"
echo "2. Run './test-deployment.sh' to verify everything works"
echo ""

print_info "Service URLs:"
echo "- Frontend: http://localhost:3000"
echo "- Backend: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo ""

print_info "For production deployment:"
echo "docker compose --profile production up --build -d"
echo ""

print_info "For WSL2 users:"
echo "Make sure to configure port forwarding if accessing from Windows"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    print_info "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    print_info "✅ .env file created. Please review and modify as needed."
fi

echo "Ready to deploy! 🚀"
