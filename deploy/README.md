# Docker 部署指南

本文档介绍如何使用 Docker Compose 部署前后端应用。

## 目录结构

```
deploy/
├── docker-compose.yml      # Docker Compose 配置
├── .env.example           # 环境变量示例
├── nginx/
│   └── nginx.conf         # Nginx 反向代理配置
└── README.md             # 部署说明
```

## 快速开始

### 1. 环境准备

确保系统已安装：
- Docker (>= 20.10)
- Docker Compose (>= 2.0)

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env
```

### 3. 启动服务

#### 开发环境（仅前后端）
```bash
docker-compose up --build
```

#### 生产环境（包含反向代理）
```bash
docker-compose --profile production up --build
```

#### 后台运行
```bash
# 开发环境
docker-compose up --build -d

# 生产环境
docker-compose --profile production up --build -d
```

### 4. 访问应用

- **开发环境**：
  - 前端: http://localhost:3000
  - 后端 API: http://localhost:8000
  - API 文档: http://localhost:8000/docs

- **生产环境**：
  - 应用: http://localhost
  - HTTPS: https://localhost (需要配置 SSL 证书)

## 服务说明

### Backend (backend-app)
- **基础镜像**: python:3.9-slim
- **多阶段构建**: 优化镜像大小
- **非 root 用户**: 提升安全性
- **健康检查**: 自动监控服务状态
- **端口**: 8000
- **环境变量**: 通过 .env 文件配置

### Frontend (frontend-app)
- **基础镜像**: nginx:alpine
- **静态资源**: 由 Nginx 托管
- **API 代理**: 自动转发 /api/ 请求到后端
- **端口**: 3000
- **健康检查**: 监控服务可用性

### Reverse Proxy (nginx)
- **基础镜像**: nginx:alpine
- **负载均衡**: 支持多实例部署
- **SSL 终止**: HTTPS 支持（需配置证书）
- **速率限制**: 防止 API 滥用
- **端口**: 80, 443

## 环境变量配置

### 必需配置
```bash
# 应用端口
BACKEND_PORT=8000
FRONTEND_PORT=3000

# 后端配置
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-change-in-production
```

### 可选配置
```bash
# 数据库
DATABASE_URL=sqlite:///./app.db

# 模型配置
MODEL_NAME=your-model-name
MODEL_PATH=/app/models

# API 密钥
API_KEY=your-api-key
```

## 数据持久化

### 日志目录
```bash
# 日志文件将保存在
./logs/
```

### 数据目录
```bash
# 应用数据将保存在
./data/
```

## 常用命令

### 查看服务状态
```bash
docker-compose ps
```

### 查看日志
```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs backend
docker-compose logs frontend
```

### 重启服务
```bash
# 重启所有服务
docker-compose restart

# 重启特定服务
docker-compose restart backend
```

### 停止服务
```bash
# 停止并删除容器
docker-compose down

# 停止但保留容器
docker-compose stop
```

### 更新镜像
```bash
# 重新构建并启动
docker-compose up --build --force-recreate
```

## WSL2 特殊配置

在 WSL2 环境中，可能需要以下额外配置：

### 1. 端口转发
确保 WSL2 端口映射正确：
```bash
# 在 Windows PowerShell 中执行（管理员权限）
netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=$(wsl hostname -I)
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$(wsl hostname -I)
```

### 2. 防火墙配置
在 Windows 防火墙中开放相应端口。

### 3. Docker Desktop 配置
确保 Docker Desktop 的 WSL2 集成已启用。

## 生产环境部署

### 1. SSL 证书配置

```bash
# 创建 SSL 目录
mkdir -p nginx/ssl

# 放置证书文件
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem
```

### 2. 环境变量优化
```bash
DEBUG=false
LOG_LEVEL=WARNING
SECRET_KEY=your-production-secret-key
```

### 3. 资源限制
在 docker-compose.yml 中添加资源限制：
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

## 故障排除

### 1. 端口冲突
如果端口被占用，修改 .env 文件中的端口配置。

### 2. 权限问题
确保 Docker 有足够权限访问项目目录。

### 3. 网络问题
```bash
# 重置 Docker 网络
docker network prune
docker-compose down
docker-compose up --build
```

### 4. 内存不足
在 WSL2 中增加内存限制：
```bash
# 在 %USERPROFILE%\.wslconfig 中添加
[wsl2]
memory=4GB
```

## 监控和日志

### 健康检查
所有服务都配置了健康检查，可以通过以下命令查看状态：
```bash
docker-compose exec backend curl -f http://localhost:8000/api/v1/health/health
docker-compose exec frontend wget --spider http://localhost:3000/health
```

### 日志聚合
生产环境建议使用 ELK Stack 或类似工具进行日志聚合。

## 安全建议

1. **定期更新基础镜像**
2. **使用强密码和密钥**
3. **启用 HTTPS**
4. **配置防火墙规则**
5. **定期备份**
6. **监控异常访问**

## 支持

如有问题，请检查：
1. Docker 和 Docker Compose 版本
2. 端口占用情况
3. 环境变量配置
4. 系统资源使用情况
