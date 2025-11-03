# Backend

基于 FastAPI 的后端应用。

## 技术栈

- Python >= 3.9
- FastAPI
- Poetry (依赖管理)
- Pytest (测试框架)
- Uvicorn (ASGI 服务器)

## 项目结构

```
backend/
├── app/                    # 应用主目录
│   ├── core/              # 核心模块
│   │   ├── config.py      # 配置管理
│   │   ├── dependencies.py # 依赖注入
│   │   └── middleware.py  # 中间件
│   ├── routers/           # 路由模块
│   │   ├── __init__.py    # 路由注册
│   │   └── health.py      # 健康检查路由
│   ├── services/          # 业务逻辑层
│   │   └── health.py      # 健康检查服务
│   ├── schemas/           # 数据模型
│   │   └── common.py      # 通用模型
│   ├── main.py           # 应用入口
│   └── __init__.py       # 包初始化
├── tests/                # 测试文件
│   ├── __init__.py       # 测试包初始化
│   └── test_main.py      # 主应用测试
├── .env.example          # 环境变量示例
├── pyproject.toml        # Poetry 配置和依赖
└── README.md            # 后端说明
```

## 开发环境

### 环境要求

- Python >= 3.9
- Poetry (推荐) 或 pip

### 安装依赖

```bash
# 使用 Poetry (推荐)
poetry install

# 或使用 pip
pip install -e .
```

### 配置环境变量

```bash
# 复制环境变量示例文件
cp .env.example .env

# 根据需要修改 .env 文件中的配置
```

### 开发环境启动

```bash
# 使用 Poetry (推荐)
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 或直接使用 uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 运行测试

```bash
# 使用 Poetry (推荐)
poetry run pytest

# 或直接使用 pytest
pytest
```

## API 文档

启动应用后，可以通过以下地址访问 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 主要接口

### 健康检查

- `GET /api/v1/health/health` - 应用健康检查
- `GET /api/v1/health/info` - 应用信息
- `GET /` - 根路径（重定向信息）

## 开发指南

### 代码规范

项目使用以下工具确保代码质量：

- **Black**: 代码格式化
- **isort**: 导入排序
- **flake8**: 代码检查
- **mypy**: 类型检查

```bash
# 格式化代码
poetry run black app/ tests/
poetry run isort app/ tests/

# 代码检查
poetry run flake8 app/ tests/
poetry run mypy app/
```

### 添加新路由

1. 在 `app/routers/` 目录下创建新的路由文件
2. 在 `app/routers/__init__.py` 中注册新路由
3. 在 `app/services/` 中实现业务逻辑
4. 在 `app/schemas/` 中定义数据模型

### 日志和异常处理

- 所有请求都会记录日志
- 统一的异常处理，返回标准格式的错误信息
- 支持不同级别的日志配置

## 部署

### 生产环境启动

```bash
# 使用 Gunicorn + Uvicorn
poetry run gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker 部署

可以创建 Dockerfile 进行容器化部署。

## 配置说明

主要配置项（通过环境变量设置）：

- `DEBUG`: 调试模式 (true/false)
- `LOG_LEVEL`: 日志级别 (DEBUG/INFO/WARNING/ERROR)
- `HOST`: 服务器地址
- `PORT`: 服务器端口
- `API_PREFIX`: API 前缀路径