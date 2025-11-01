# Docker Deployment Guide

## Overview
This guide covers Docker and Docker Compose deployment for the Transformer Visualization platform.

## Fixed Issues

### 1. Missing Design System Module
**Problem**: Turbopack build failed with "Module not found: Can't resolve '@/lib/design-system'"
**Solution**: Created `/src/lib/design-system.ts` with:
- `cn` utility function using clsx + tailwind-merge
- `colors` object with neutral, module-specific, and visualization colors

### 2. TypeScript Type Issues
**Problem**: Various TypeScript errors in components
**Solution**: 
- Fixed `unknown` type conditional rendering in ExportToolbar
- Updated MotionValue types in ScrollytellingLayout

### 3. Google Fonts Network Issues
**Problem**: Docker build failed due to Google Fonts network connectivity
**Solution**: Removed Google Fonts imports and used system fonts instead

### 4. Docker Build Optimization
**Problem**: Multi-stage build issues with npm ci vs npm install
**Solution**: Used npm install for better compatibility

## Deployment Options

### Frontend-Only Deployment (Recommended)
For environments where backend is not required or will be deployed separately:

```bash
# Build and run frontend only
docker compose -f docker-compose.frontend.yml up --build -d

# Access the application
http://localhost:3000
```

### Full Stack Deployment (When Network Issues Resolved)
For complete application with backend:

```bash
# Build and run full stack
docker compose up --build -d

# Access services
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

## Configuration

### Environment Variables
- `NODE_ENV=production` - Production mode
- `NEXT_PUBLIC_API_URL=http://localhost:8000` - Backend API URL

### Ports
- Frontend: 3000
- Backend: 8000

## Troubleshooting

### Build Failures
1. **Network Issues**: If builds fail due to network connectivity, use frontend-only deployment
2. **Module Resolution**: Ensure `@/lib/design-system.ts` exists and is properly exported
3. **TypeScript Errors**: Run `npm run type-check` locally to identify issues

### Runtime Issues
1. **Port Conflicts**: Ensure ports 3000 and 8000 are available
2. **Memory**: Docker requires sufficient memory for Node.js builds
3. **Permissions**: Ensure Docker daemon has proper permissions

### Health Checks
Backend health checks are disabled due to network dependency issues. They can be re-enabled when:
- Network connectivity is stable
- curl is installed in backend container

## Development vs Production

### Development
```bash
cd frontend
npm install
npm run dev
```

### Production (Docker)
```bash
docker compose -f docker-compose.frontend.yml up --build -d
```

## Image Sizes
- Frontend: ~193MB (Alpine Linux based)
- Backend: ~500MB+ (Python slim based)

## Next Steps
1. Resolve backend network connectivity issues
2. Enable health checks
3. Add environment-specific configurations
4. Implement CI/CD pipeline
5. Add monitoring and logging