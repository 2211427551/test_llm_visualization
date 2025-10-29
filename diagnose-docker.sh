#!/bin/bash
# Docker部署诊断脚本
# 用于检查和诊断Docker部署中的常见问题

echo "=================================="
echo "Docker部署诊断脚本"
echo "=================================="
echo ""

# 检查Docker是否安装
echo "1. 检查Docker安装..."
if command -v docker &> /dev/null; then
    docker --version
    echo "✓ Docker已安装"
else
    echo "✗ Docker未安装！请先安装Docker"
    exit 1
fi
echo ""

# 检查Docker Compose是否安装
echo "2. 检查Docker Compose安装..."
if command -v docker compose &> /dev/null; then
    docker compose version
    echo "✓ Docker Compose已安装"
else
    echo "✗ Docker Compose未安装！请先安装Docker Compose"
    exit 1
fi
echo ""

# 检查Docker服务是否运行
echo "3. 检查Docker服务状态..."
if docker info &> /dev/null; then
    echo "✓ Docker服务正在运行"
else
    echo "✗ Docker服务未运行！请启动Docker"
    exit 1
fi
echo ""

# 检查容器运行状态
echo "4. 检查容器运行状态..."
docker compose ps
echo ""

# 检查端口映射
echo "5. 检查端口映射..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# 测试后端健康检查
echo "6. 测试后端健康检查..."
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo "✓ 后端健康检查通过"
    curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
else
    echo "✗ 后端健康检查失败"
    echo "  尝试访问根路径..."
    curl -s http://localhost:8000/ | jq . 2>/dev/null || curl -s http://localhost:8000/
fi
echo ""

# 测试后端API健康检查
echo "7. 测试后端API健康检查..."
if curl -f -s http://localhost:8000/api/health > /dev/null; then
    echo "✓ 后端API健康检查通过"
    curl -s http://localhost:8000/api/health | jq . 2>/dev/null || curl -s http://localhost:8000/api/health
else
    echo "✗ 后端API健康检查失败"
fi
echo ""

# 测试前端访问
echo "8. 测试前端访问..."
if curl -f -s http://localhost:3000 > /dev/null 2>&1; then
    echo "✓ 前端可访问"
else
    echo "✗ 前端不可访问"
    echo "  检查前端容器日志..."
    docker compose logs --tail=20 frontend
fi
echo ""

# 检查网络配置
echo "9. 检查Docker网络..."
docker network ls | grep transformer-viz
echo ""

# 检查容器内部的前端文件
echo "10. 检查前端容器内部..."
if docker compose ps | grep -q "frontend.*running"; then
    echo "前端容器正在运行，检查内部文件..."
    docker compose exec -T frontend ls -la /app 2>/dev/null || echo "无法访问前端容器"
else
    echo "前端容器未运行"
fi
echo ""

# 显示最近的日志
echo "11. 最近的容器日志..."
echo "--- 后端日志 (最后10行) ---"
docker compose logs --tail=10 backend
echo ""
echo "--- 前端日志 (最后10行) ---"
docker compose logs --tail=10 frontend
echo ""

# WSL环境检测
echo "12. WSL环境检测..."
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "✓ 检测到WSL环境"
    echo "  WSL IP地址："
    hostname -I
    echo ""
    echo "  如果从Windows浏览器无法访问localhost:3000，"
    echo "  请尝试使用上述IP地址：http://<IP>:3000"
else
    echo "未检测到WSL环境"
fi
echo ""

echo "=================================="
echo "诊断完成"
echo "=================================="
echo ""
echo "如果发现问题，请尝试："
echo "1. 重启容器：docker compose down && docker compose up --build"
echo "2. 清理并重建：docker compose down -v && docker compose up --build"
echo "3. 查看详细日志：docker compose logs -f"
echo ""
