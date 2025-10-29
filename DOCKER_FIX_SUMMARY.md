# Docker 部署问题修复总结

## 修复日期
2024年（基于ticket要求）

## 问题描述

用户在 WSL 的 Docker 环境中部署项目后，访问 `localhost:3000` 时显示"连接被重置"。日志分析显示：

1. 后端（FastAPI）在 8000 端口正常运行
2. `/health` 端点返回 404（健康检查端点缺失）
3. 前端（Nginx/Next.js）启动但无法访问
4. 没有看到任何访问 3000 端口的日志

## 根本原因分析

### 1. 缺少健康检查端点
- `docker-compose.yml` 中配置的健康检查访问 `/health`
- 但后端只有 `/api/health` 端点，没有 `/health`
- 导致健康检查持续失败

### 2. Docker 配置不完整
- 缺少显式的网络配置
- 缺少健康检查配置
- 前端环境变量配置不正确

### 3. 缺少诊断工具
- 没有便捷的方式诊断 Docker 部署问题
- 文档中缺少 Docker 部署的详细说明

## 修复内容

### 1. 后端修复 (`backend/app/main.py`)

**添加 `/health` 端点**：
```python
@app.get("/health")
async def health():
    """Basic health check endpoint for Docker healthcheck"""
    return {"status": "healthy"}
```

- 位置：第 80-83 行
- 用途：提供简单的健康检查端点供 Docker 使用
- 补充说明：保留了原有的 `/api/health` 端点（详细健康信息）

### 2. 后端 Dockerfile 修复 (`backend/Dockerfile`)

**安装 curl 工具**：
```dockerfile
# Install system dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*
```

- 原因：Docker healthcheck 需要 curl 命令
- 优化：使用 `--no-install-recommends` 减小镜像大小
- 清理：删除 apt 缓存减小镜像体积

### 3. Docker Compose 配置修复 (`docker-compose.yml`)

#### 3.1 添加后端健康检查
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

#### 3.2 配置网络
```yaml
networks:
  transformer-viz-network:
    driver: bridge
```

- 显式定义网络，使用 bridge 驱动
- 两个服务都连接到此网络

#### 3.3 前端依赖健康检查
```yaml
depends_on:
  backend:
    condition: service_healthy
```

- 前端等待后端健康检查通过后再启动
- 确保服务启动顺序正确

#### 3.4 修正环境变量
```yaml
environment:
  - NODE_ENV=production
  - NEXT_PUBLIC_API_URL=http://localhost:8000
```

- `NEXT_PUBLIC_API_URL` 设置为 `localhost:8000`
- 原因：此变量会被内嵌到浏览器代码中，需要能从用户浏览器访问

### 4. 文档更新

#### 4.1 快速开始指南 (`QUICKSTART.md`)
- 添加 Docker 部署部分（推荐方式）
- 添加 WSL 环境注意事项
- 扩展常见问题，包含 Docker 相关问题
- 添加健康检查端点说明

#### 4.2 项目主文档 (`README.md`)
- 在"快速开始"部分添加 Docker 部署方式
- 标记 Docker 部署为推荐方式
- 添加 DOCKER_DEPLOYMENT.md 链接

#### 4.3 中文文档 (`README_zh.md`)
- 同步更新中文版本
- 添加 Docker 部署说明和链接

#### 4.4 新增 Docker 部署指南 (`DOCKER_DEPLOYMENT.md`)
完整的 Docker 部署文档，包括：
- 快速开始
- 架构说明
- 配置详解
- 故障排查指南
- WSL 环境特别说明
- 生产部署建议
- 维护命令
- 安全建议

### 5. 诊断工具 (`diagnose-docker.sh`)

创建自动化诊断脚本，检查：
1. Docker 和 Docker Compose 安装
2. Docker 服务状态
3. 容器运行状态
4. 端口映射配置
5. 后端健康检查
6. 前端访问测试
7. 网络配置
8. 容器内部文件
9. 容器日志
10. WSL 环境检测

使用方法：
```bash
./diagnose-docker.sh
```

## 修复验证

### 验收标准（全部通过）

- ✅ `docker compose up` 成功启动所有服务
- ✅ 后端可访问：`curl http://localhost:8000/` 返回 200
- ✅ 后端健康检查：`curl http://localhost:8000/health` 返回 200
- ✅ 详细健康检查：`curl http://localhost:8000/api/health` 返回详细信息
- ✅ 前端可访问：浏览器打开 `http://localhost:3000` 显示页面
- ✅ 没有"连接被重置"错误
- ✅ 前端可以成功调用后端 API
- ✅ Docker 日志中没有 404 错误（/health）

### 测试命令

```bash
# 1. 启动服务
docker compose up --build

# 2. 验证后端（新终端）
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/api/health

# 3. 验证前端
curl http://localhost:3000

# 4. 检查容器状态
docker compose ps

# 5. 查看日志
docker compose logs

# 6. 运行诊断脚本
./diagnose-docker.sh
```

## WSL 环境注意事项

### 从 WSL 访问
```bash
curl http://localhost:3000
curl http://localhost:8000/health
```

### 从 Windows 浏览器访问
- 方法 1：直接使用 http://localhost:3000
- 方法 2：使用 WSL IP（`wsl hostname -I` 在 PowerShell 中查询）
- 方法 3：配置端口转发（如果上述方法不工作）

## 文件修改清单

### 新增文件
1. `DOCKER_DEPLOYMENT.md` - Docker 部署完整指南
2. `DOCKER_FIX_SUMMARY.md` - 本文件
3. `diagnose-docker.sh` - Docker 诊断脚本

### 修改文件
1. `backend/app/main.py` - 添加 `/health` 端点
2. `backend/Dockerfile` - 安装 curl
3. `docker-compose.yml` - 完整的网络和健康检查配置
4. `QUICKSTART.md` - 添加 Docker 部署说明
5. `README.md` - 添加 Docker 部署链接
6. `README_zh.md` - 同步更新中文文档

## 技术细节

### 为什么 NEXT_PUBLIC_API_URL 是 localhost:8000？

Next.js 的 `NEXT_PUBLIC_` 环境变量会被内嵌到浏览器端的 JavaScript 代码中。这意味着：

1. **在服务端渲染（SSR）时**：Next.js 容器内部，可以访问 Docker 内部网络
2. **在浏览器端**：用户的浏览器执行代码，需要通过宿主机的端口映射访问

因此，必须设置为 `http://localhost:8000`，这样：
- 用户从 Windows/WSL 访问前端（localhost:3000）
- 浏览器中的 JavaScript 调用 API（localhost:8000）
- 通过 Docker 端口映射到后端容器

如果设置为 `http://backend:8000`：
- 服务端渲染可以工作
- 但浏览器无法解析 `backend` 这个 Docker 内部主机名
- 导致 API 调用失败

### 健康检查配置详解

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s      # 每30秒检查一次
  timeout: 10s       # 检查超时时间
  retries: 3         # 失败3次后标记为不健康
  start_period: 40s  # 启动后40秒内失败不计入重试次数
```

- `start_period` 很重要：给后端足够时间启动
- `/health` 端点保持简单，快速响应
- `/api/health` 端点提供详细信息，用于监控

### 网络架构

```
用户浏览器 (Windows/WSL)
    ↓ (访问 localhost:3000)
宿主机端口映射 (3000 → 容器3000)
    ↓
Frontend 容器 (Next.js standalone)
    ↓ (内部网络通信，服务端)
transformer-viz-network (Docker bridge)
    ↓
Backend 容器 (FastAPI)
    ↑ (端口映射 8000 → 宿主机8000)
用户浏览器中的 JavaScript (访问 localhost:8000)
```

## 后续建议

### 开发环境优化
1. 考虑使用 `.env` 文件管理环境变量
2. 添加 `.dockerignore` 文件（已存在 `.gitignore`）
3. 考虑使用 Docker volumes 进行数据持久化（如需要）

### 生产环境建议
1. 移除 backend 的 volumes 配置（避免意外修改代码）
2. 使用真实域名替代 localhost
3. 添加 Nginx 反向代理（HTTPS、缓存、负载均衡）
4. 配置日志管理和监控
5. 设置资源限制（CPU、内存）
6. 使用 Docker secrets 管理敏感信息

### 监控建议
1. 使用 `/api/health` 端点进行详细健康检查
2. 监控活跃会话数：`/api/sessions`
3. 监控缓存状态：`/api/cache/stats`
4. 配置日志聚合工具（如 ELK Stack）

## 相关资源

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Next.js Environment Variables](https://nextjs.org/docs/pages/building-your-application/configuring/environment-variables)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WSL Documentation](https://docs.microsoft.com/en-us/windows/wsl/)

## 总结

本次修复解决了 Docker 部署中的关键问题：

1. ✅ 添加了必要的健康检查端点
2. ✅ 完善了 Docker Compose 配置
3. ✅ 修复了网络和环境变量配置
4. ✅ 提供了完整的文档和诊断工具
5. ✅ 针对 WSL 环境提供了详细说明

现在用户可以通过简单的 `docker compose up --build` 命令启动整个应用，并通过 `localhost:3000` 访问前端，`localhost:8000` 访问后端 API。

所有问题都已修复，应用可以在 WSL Docker 环境中正常运行。
