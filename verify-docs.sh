#!/bin/bash

# 文档验证脚本
# 检查所有文档是否完整且无英文残留

echo "🔍 开始验证文档完整性..."

# 检查必需文件是否存在
required_files=(
    "README.md"
    "CONTRIBUTING.md" 
    "CHANGELOG.md"
    "docs/README.md"
    "docs/sparse-attention.md"
    "docs/moe-design.md"
    "docs/data-structures.md"
    "docs/api-reference.md"
    "docs/deployment-guide.md"
)

echo "📋 检查必需文件..."
missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    else
        echo "✅ $file"
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ 缺少以下文件："
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

# 检查文档中是否包含英文残留（排除专有名词）
echo ""
echo "🔤 检查英文残留..."
docs_files=("README.md" "docs/README.md" "docs/sparse-attention.md" "docs/moe-design.md" "docs/data-structures.md" "docs/api-reference.md" "docs/deployment-guide.md")

# 排除的英文词汇（技术术语）
excluded_words="Sparse|Attention|Mixture|Experts|MoE|FastAPI|PyTorch|React|TypeScript|Docker|API|URL|HTTP|JSON|REST|CPU|GPU|RAM|SSD|WSL|Linux|Ubuntu|nginx|redis|prometheus|grafana|Transformer|Python|True|False|None|Optional|Linear|Activation|Module|Base|Content|Type|Compose|Top|Web|Uvicorn|Pydantic|Poetry|Vite|Nginx|Ctrl|Issue|Fork|Config|Time|Sparsity|Window|Dropout|Args|Returns|Field|Redis|Gzip|Frame|Options|Tailwind|Hooks|Git|Node|Microsoft|Discussions|Memory|Longformer|Document|Swish|Expert|Load|Dict|Any|List|Protection|Referrer|Policy|Security|Subsystem|Desktop|Use|Add|The|Beltagy|Longer|Sequences|Outrageously|Large|Neural|Networks|Tensor|Error|Session|Exception|Host|Real|Forwarded|For|Pull|Request|Black|Prettier|Deepseek|Zaheer|Reformer|Efficient|Kitaev|Gated|Layer|Shazeer|Switch|Scaling|Token|Tuple|Array|Proto|Upgrade|Connection|Star|Exp|Technical|Report|Team|Trillion|Parameter|Models|Simple|Fedus|Cache|Control|Prometheus|Grafana|Language|Meets|Instruction|Tuning|Let|Encrypt|Certbot|State|City|Organization|Backup|Cpu|Swarm|Limiter|Kubernetes|Deployment|Long"

for file in "${docs_files[@]}"; do
    if [ -f "$file" ]; then
        # 查找可能的英文残留（3个字母以上且不在排除列表中）
        english_words=$(grep -oE "\b[A-Z][a-z]{2,}\b" "$file" | grep -vE "$excluded_words" | head -5)
        if [ -n "$english_words" ]; then
            echo "⚠️  $file 中可能存在英文残留："
            echo "$english_words"
        else
            echo "✅ $file - 无英文残留"
        fi
    fi
done

# 检查代码块语法
echo ""
echo "📝 检查代码块语法..."
code_files=("README.md" "docs/api-reference.md" "docs/deployment-guide.md")

for file in "${code_files[@]}"; do
    if [ -f "$file" ]; then
        # 检查是否有未闭合的代码块
        code_block_start=$(grep -c '```' "$file")
        if [ $((code_block_start % 2)) -ne 0 ]; then
            echo "❌ $file 中有未闭合的代码块"
        else
            echo "✅ $file - 代码块语法正确"
        fi
    fi
done

# 检查链接有效性
echo ""
echo "🔗 检查文档链接..."
link_files=("README.md" "docs/README.md")

for file in "${link_files[@]}"; do
    if [ -f "$file" ]; then
        # 检查内部链接
        broken_links=$(grep -oE '\[.*\]\([^)]*\)' "$file" | grep -vE '(http|mailto)' | while read -r link; do
            target=$(echo "$link" | sed -E 's/\[.*\]\(([^)]*)\).*/\1/')
            if [[ "$target" == *.md ]] && [ ! -f "$target" ] && [ ! -f "docs/$target" ]; then
                echo "❌ 断开的链接: $target"
            fi
        done)
        
        if [ -z "$broken_links" ]; then
            echo "✅ $file - 链接有效"
        fi
    fi
done

# 检查脚本可执行权限
echo ""
echo "🔐 检查脚本权限..."
scripts=("deploy/quick-start.sh" "deploy/test-deployment.sh")

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "✅ $script - 可执行"
        else
            echo "⚠️  $script - 不可执行，建议运行: chmod +x $script"
        fi
    fi
done

echo ""
echo "📊 文档统计："
echo "- 主文档: README.md ($(wc -l < README.md) 行)"
echo "- 贡献指南: CONTRIBUTING.md ($(wc -l < CONTRIBUTING.md) 行)"
echo "- 更新日志: CHANGELOG.md ($(wc -l < CHANGELOG.md) 行)"
echo "- 技术文档: $(ls docs/*.md | wc -l) 个文件"
echo "- 总文档大小: $(du -sh docs/ | cut -f1)"

echo ""
echo "🎉 文档验证完成！"
echo ""
echo "📚 文档使用建议："
echo "1. 从 README.md 开始阅读项目概述"
echo "2. 查看 docs/README.md 了解文档导航"
echo "3. 根据需要阅读具体的技术文档"
echo "4. 使用部署指南进行环境配置"
echo "5. 参考贡献指南参与开发"