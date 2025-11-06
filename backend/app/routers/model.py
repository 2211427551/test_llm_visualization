import json
from json import JSONDecodeError
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.api import ForwardRequest, ForwardResponse, InitializeResponse
from app.services import model_inference_service

router = APIRouter()


@router.get("/initialize", response_model=InitializeResponse, summary="初始化模型")
async def initialize_model(config: Optional[str] = Query(None, description="模型配置 JSON 字符串")) -> InitializeResponse:
    """初始化或刷新模型配置。"""
    overrides: Optional[Dict[str, Any]] = None
    if config:
        try:
            parsed = json.loads(config)
        except JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="配置解析失败，请提供合法的 JSON 字符串") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="配置参数格式不正确，应为 JSON 对象")
        overrides = parsed

    try:
        payload = model_inference_service.initialize(overrides)
        return InitializeResponse(success=True, message="模型初始化成功", config=payload)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - 防御性处理
        raise HTTPException(status_code=500, detail=f"模型初始化失败: {exc}") from exc


@router.post("/forward", response_model=ForwardResponse, summary="执行前向推理")
async def forward(request: ForwardRequest) -> ForwardResponse:
    """执行一次前向推理并返回可视化数据。"""
    try:
        result = model_inference_service.forward(request.text, request.capture_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - 防御性处理
        raise HTTPException(status_code=500, detail=f"模型推理错误: {exc}") from exc

    return ForwardResponse(
        success=True,
        message="前向传播完成",
        logits_shape=result["logitsShape"],
        sequence_length=result["sequenceLength"],
        captured_data=result.get("capturedData"),
    )
