#!/bin/bash

# Docker 部署快速启动脚本
# 此脚本提供启动应用程序的便捷命令

set -e

# 输出颜色配置
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}     Docker 部署快速启动             ${NC}"
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

echo "可用命令："
echo ""

print_command "./test-deployment.sh" "构建并测试完整部署"
print_command "docker compose up --build -d" "后台启动服务"
print_command "docker compose up --build" "启动服务并显示日志"
print_command "docker compose down" "停止并移除服务"
print_command "docker compose logs -f" "跟踪日志"
print_command "docker compose ps" "显示服务状态"
echo ""

print_info "环境设置："
echo "1. 复制 .env.example 到 .env 并根据需要配置"
echo "2. 运行 './test-deployment.sh' 验证一切正常"
echo ""

print_info "服务地址："
echo "- 前端: http://localhost:3000"
echo "- 后端: http://localhost:8000"
echo "- API 文档: http://localhost:8000/docs"
echo ""

print_info "生产环境部署："
echo "docker compose --profile production up --build -d"
echo ""

print_info "WSL2 用户注意："
echo "如果从 Windows 访问，请确保配置端口转发"
echo ""

# 检查 .env 文件是否存在
if [ ! -f .env ]; then
    print_info "⚠️  未找到 .env 文件。正在从模板创建..."
    cp .env.example .env
    print_info "✅ .env 文件已创建。请检查并根据需要修改。"
fi

echo "准备部署！🚀"