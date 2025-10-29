# Docker 部署指南

本文档详细说明如何使用 Docker 和 Docker Compose 部署 Transformer 可视化应用。

## 目录

- [快速开始](#快速开始)
- [架构说明](#架构说明)
- [配置说明](#配置说明)
- [故障排查](#故障排查)
- [WSL 环境注意事项](#wsl-环境注意事项)

## 快速开始

### 前置要求

- Docker 20.10+
- Docker Compose V2
- （WSL环境）WSL 2

### 启动应用

```bash
# 克隆仓库（如果还没有）
git clone <repository-url>
cd <repository-name>

# 构建并启动所有服务
docker compose up --build

# 或在后台运行
docker compose up -d --build
```

### 访问应用

- **前端应用**：http://localhost:3000
- **后端 API**：http://localhost:8000
- **API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health

## 架构说明

### 服务组成

```
┌─────────────────────────────────────────┐
│          User's Browser                  │
│     (http://localhost:3000)              │
└────────────────┬────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────┐
│   Frontend Container (Next.js)          │
│   - Port: 3000                           │
│   - Standalone mode                      │
│   - API calls to backend                 │
└────────────────┬────────────────────────┘
                 │ Internal Network
                 │ (transformer-viz-network)
                 ↓
┌─────────────────────────────────────────┐
│   Backend Container (FastAPI)           │
│   - Port: 8000                           │
│   - Health check: /health                │
│   - CORS enabled                         │
└─────────────────────────────────────────┘
```

### 网络配置

- **网络名称**：`transformer-viz-network`
- **驱动类型**：bridge
- **服务通信**：backend 和 frontend 通过内部网络通信
- **端口映射**：
  - `3000:3000` (前端)
  - `8000:8000` (后端)

### 健康检查

后端服务配置了健康检查：
- **端点**：`/health`
- **间隔**：30秒
- **超时**：10秒
- **重试**：3次
- **启动时间**：40秒

前端服务依赖后端的健康检查通过后才会启动。

## 配置说明

### 环境变量

#### 后端环境变量

```yaml
environment:
  - ENVIRONMENT=production
```

#### 前端环境变量

```yaml
environment:
  - NODE_ENV=production
  - NEXT_PUBLIC_API_URL=http://localhost:8000
```

> **注意**：`NEXT_PUBLIC_API_URL` 设置为 `http://localhost:8000`，因为这个变量会被内嵌到浏览器端代码中，需要能从用户的浏览器访问。

### 数据卷（Volumes）

后端使用数据卷进行代码热重载（开发模式）：
```yaml
volumes:
  - ./backend:/app
```

> **生产环境建议**：移除此卷配置，避免意外修改代码。

## 故障排查

### 1. 使用诊断脚本

```bash
./diagnose-docker.sh
```

这个脚本会自动检查：
- Docker 安装状态
- 容器运行状态
- 端口映射
- 健康检查
- 网络配置
- WSL 环境

### 2. 常见问题

#### 问题：访问 localhost:3000 显示"连接被重置"

**原因**：
- 前端容器未启动
- 端口映射配置错误
- 容器内部服务未正确启动

**解决方案**：
```bash
# 1. 检查容器状态
docker compose ps

# 2. 查看前端日志
docker compose logs frontend

# 3. 检查端口映射
docker ps --format "table {{.Names}}\t{{.Ports}}"

# 4. 重启服务
docker compose restart frontend
```

#### 问题：后端健康检查失败（404错误）

**原因**：
- `/health` 端点不存在
- curl 未安装在容器中

**解决方案**：
本项目已修复此问题：
- 后端添加了 `/health` 端点
- Dockerfile 安装了 curl

#### 问题：前端无法连接后端 API

**原因**：
- 网络配置错误
- 环境变量配置错误
- CORS 配置问题

**解决方案**：
```bash
# 1. 检查网络
docker network inspect transformer-viz-network

# 2. 从前端容器测试后端连接
docker compose exec frontend wget -O- http://backend:8000/health

# 3. 检查 CORS 配置（后端日志）
docker compose logs backend | grep -i cors
```

#### 问题：Docker Compose 版本警告

**警告信息**：
```
WARN[0000] /path/to/docker-compose.yml: `version` is obsolete
```

**说明**：
这是正常的警告信息。Docker Compose V2 不再需要 `version` 字段，但为了向后兼容，我们保留了它。可以安全忽略此警告。

### 3. 查看日志

```bash
# 所有服务日志
docker compose logs -f

# 特定服务日志
docker compose logs -f backend
docker compose logs -f frontend

# 最近的日志（最后100行）
docker compose logs --tail=100
```

### 4. 进入容器调试

```bash
# 进入后端容器
docker compose exec backend bash

# 进入前端容器
docker compose exec frontend sh

# 检查前端构建产物
docker compose exec frontend ls -la /app

# 检查后端健康状态
docker compose exec backend curl http://localhost:8000/health
```

### 5. 完全重建

```bash
# 停止并删除所有容器、网络、卷
docker compose down -v

# 清理未使用的镜像
docker image prune -a

# 重新构建并启动
docker compose up --build
```

## WSL 环境注意事项

### WSL 2 网络特性

在 WSL 2 环境中，网络配置有一些特殊之处：

#### 从 WSL 内部访问

```bash
# 在 WSL 终端中
curl http://localhost:3000
curl http://localhost:8000/health
```

这应该正常工作。

#### 从 Windows 访问 WSL 服务

**方法 1：使用 localhost（推荐）**

在 Windows 11 或较新的 Windows 10 版本中，可以直接使用 localhost：
- http://localhost:3000
- http://localhost:8000

**方法 2：使用 WSL IP 地址**

如果方法 1 不工作：

1. 在 WSL 中查找 IP 地址：
   ```bash
   hostname -I
   # 或
   ip addr show eth0 | grep inet
   ```

2. 在 Windows 浏览器中访问：
   ```
   http://<WSL_IP>:3000
   ```

**方法 3：配置端口转发**

如果上述方法都不工作，可以在 Windows PowerShell（管理员）中配置端口转发：

```powershell
# 查找 WSL IP
wsl hostname -I

# 添加端口转发规则
netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=<WSL_IP>
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<WSL_IP>

# 查看端口转发规则
netsh interface portproxy show all

# 删除端口转发规则（如果需要）
netsh interface portproxy delete v4tov4 listenport=3000 listenaddress=0.0.0.0
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0
```

### WSL 2 性能优化

1. **使用 WSL 2 文件系统**：
   - 将项目克隆到 WSL 文件系统中（如 `/home/user/projects`）
   - 避免使用 Windows 文件系统（如 `/mnt/c/...`）
   - 这样可以大幅提升 Docker 性能

2. **配置 Docker Desktop**：
   - 确保 Docker Desktop 设置中启用了 WSL 2 集成
   - 分配足够的内存和 CPU 资源

3. **网络优化**：
   - 在 `.wslconfig` 文件中配置网络设置：
     ```ini
     [wsl2]
     memory=4GB
     processors=2
     localhostForwarding=true
     ```

## 生产部署建议

### 1. 移除开发用的数据卷

编辑 `docker-compose.yml`，移除后端的 volumes 配置：

```yaml
backend:
  # 移除或注释掉以下行
  # volumes:
  #   - ./backend:/app
```

### 2. 使用环境文件

创建 `.env` 文件：

```env
# Backend
ENVIRONMENT=production
CORS_ORIGINS=https://yourdomain.com

# Frontend
NODE_ENV=production
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

### 3. 添加反向代理

使用 Nginx 或 Traefik 作为反向代理：
- 处理 HTTPS/SSL
- 负载均衡
- 静态文件缓存

### 4. 配置日志管理

```yaml
backend:
  logging:
    driver: "json-file"
    options:
      max-size: "10m"
      max-file: "3"
```

### 5. 限制资源使用

```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '1'
        memory: 1G
      reservations:
        cpus: '0.5'
        memory: 512M
```

## 维护命令

```bash
# 查看资源使用情况
docker stats

# 清理未使用的资源
docker system prune -a

# 查看镜像大小
docker images

# 导出日志
docker compose logs > app-logs.txt

# 备份数据（如果有持久化数据）
docker compose down
tar -czf backup.tar.gz ./data

# 更新服务
git pull
docker compose up --build -d
```

## 安全建议

1. **不要在生产环境中暴露 8000 端口**
2. **使用 secrets 管理敏感信息**
3. **定期更新基础镜像**
4. **启用 Docker 内容信任**
5. **使用非 root 用户运行容器**（前端已配置）

## 获取帮助

如果遇到问题：
1. 运行诊断脚本：`./diagnose-docker.sh`
2. 查看日志：`docker compose logs -f`
3. 检查 GitHub Issues
4. 查看 [Docker 文档](https://docs.docker.com/)
5. 查看 [WSL 文档](https://docs.microsoft.com/en-us/windows/wsl/)

## 相关文档

- [快速开始指南](./QUICKSTART.md)
- [前端 README](./frontend/README.md)
- [后端 README](./backend/README.md)
