from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from app.models.request import InitRequest, StepRequest, TraceRequest, ChunkRequest, WebSocketMessage
from app.models.response import (
    InitResponse, StepResponse, InitialState, TraceResponse, 
    ChunkedResponse, StreamMessage, SparsityInfo, MoERouting
)
from app.services.tokenizer import SimpleTokenizer
from app.services.embedding import EmbeddingLayer
from app.services.transformer import TransformerSimulator
from app.services.sparse_attention import SparseAttentionSimulator
from app.services.moe import MoESimulator
import uuid
from typing import Dict, Any, AsyncGenerator, List
import numpy as np
from functools import lru_cache
import hashlib
import json
import asyncio
import time
from contextlib import asynccontextmanager


app = FastAPI(
    title="Transformer计算模拟器API - 扩展版",
    description="支持MoE、稀疏注意力、流式传输和可视化的Transformer计算模拟器后端",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression middleware for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

sessions: Dict[str, Dict[str, Any]] = {}
computation_cache: Dict[str, Any] = {}  # LRU cache for computation results
websocket_connections: Dict[str, WebSocket] = {}  # Active WebSocket connections


def numpy_to_list(arr: np.ndarray, precision: int = 4) -> list:
    """Convert numpy array to list with configurable precision for smaller JSON size"""
    if isinstance(arr, np.ndarray):
        # Round to specified precision to reduce data size
        return np.round(arr, precision).tolist()
    return arr


def generate_cache_key(text: str, config: dict) -> str:
    """Generate a unique cache key based on input text and config"""
    config_str = json.dumps(config, sort_keys=True)
    key_str = f"{text}:{config_str}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_computation(cache_key: str) -> Any:
    """Retrieve cached computation result"""
    return computation_cache.get(cache_key)


def cache_computation(cache_key: str, result: Any) -> None:
    """Cache computation result with simple LRU (max 100 entries)"""
    if len(computation_cache) >= 100:
        # Remove oldest entry (simple FIFO, not true LRU but good enough)
        first_key = next(iter(computation_cache))
        del computation_cache[first_key]
    computation_cache[cache_key] = result


@app.get("/")
async def root():
    return {
        "message": "Transformer计算模拟器API",
        "docs": "/docs",
        "endpoints": {
            "init": "/api/init",
            "step": "/api/step"
        }
    }


@app.get("/health")
async def health():
    """Basic health check endpoint for Docker healthcheck"""
    return {"status": "healthy"}


@app.post("/api/init", response_model=InitResponse)
async def initialize_session(request: InitRequest):
    try:
        request.config.validate_config()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="输入文本不能为空")
    
    # Generate cache key
    cache_key = generate_cache_key(
        request.text, 
        request.config.model_dump()
    )
    
    # Check cache first
    cached_result = get_cached_computation(cache_key)
    
    if cached_result:
        # Use cached result but create a new session
        session_id = str(uuid.uuid4())
        sessions[session_id] = cached_result["session_data"]
        
        response = InitResponse(
            session_id=session_id,
            tokens=cached_result["tokens"],
            token_texts=cached_result["token_texts"],
            total_steps=cached_result["total_steps"],
            initial_state=InitialState(
                embeddings=cached_result["embeddings"],
                positional_encodings=cached_result["positional_encodings"]
            ),
            sparse_info=cached_result.get("sparse_info"),
            moe_info=cached_result.get("moe_info")
        )
        return response
    
    # Compute if not cached
    session_id = str(uuid.uuid4())
    
    tokenizer = SimpleTokenizer(request.config.n_vocab)
    token_ids, token_texts = tokenizer.tokenize(request.text)
    
    if len(token_ids) == 0:
        raise HTTPException(status_code=400, detail="分词结果为空")
    
    if len(token_ids) > request.config.max_seq_len:
        raise HTTPException(
            status_code=400, 
            detail=f"序列长度 {len(token_ids)} 超过最大长度 {request.config.max_seq_len}"
        )
    
    embedding_layer = EmbeddingLayer(
        request.config.n_vocab,
        request.config.n_embd,
        request.config.max_seq_len
    )
    
    embeddings, positional_encodings = embedding_layer.embed(token_ids)
    initial_input = embedding_layer.get_initial_representation(token_ids)
    
    # Get precision from config
    precision = request.config.viz_config.precision if request.config.viz_config else 4
    
    transformer = TransformerSimulator(
        request.config.n_embd,
        request.config.n_layer,
        request.config.n_head,
        request.config
    )
    
    computation_steps = transformer.simulate(initial_input)
    
    # Generate sparse info if sparse attention is enabled
    sparse_info = None
    if request.config.attention_type == "sparse" and request.config.sparse_config:
        from app.services.sparse_attention import SparseAttentionMask
        mask_generator = SparseAttentionMask(request.config.sparse_config, len(token_ids))
        attention_mask, sparsity_info = mask_generator.generate_mask()
        sparse_info = sparsity_info
    
    # Generate MoE info if MoE is enabled
    moe_info = None
    if request.config.moe_config and request.config.moe_config.enabled:
        moe_info = {
            "enabled": True,
            "n_experts": request.config.moe_config.n_experts,
            "top_k": request.config.moe_config.top_k,
            "gate_noise": request.config.moe_config.gate_noise,
            "gate_dropout": request.config.moe_config.gate_dropout
        }
    
    session_data = {
        "config": request.config,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "embeddings": embeddings,
        "positional_encodings": positional_encodings,
        "initial_input": initial_input,
        "computation_steps": computation_steps,
        "tokenizer": tokenizer,
        "sparse_info": sparse_info,
        "moe_info": moe_info
    }
    
    sessions[session_id] = session_data
    
    total_steps = len(computation_steps)
    
    # Prepare response with configurable precision
    embeddings_list = numpy_to_list(embeddings, precision)
    positional_encodings_list = numpy_to_list(positional_encodings, precision)
    
    response = InitResponse(
        session_id=session_id,
        tokens=token_ids,
        token_texts=token_texts,
        total_steps=total_steps,
        initial_state=InitialState(
            embeddings=embeddings_list,
            positional_encodings=positional_encodings_list
        ),
        sparse_info=sparse_info,
        moe_info=moe_info
    )
    
    # Cache the result
    cache_computation(cache_key, {
        "session_data": session_data,
        "tokens": token_ids,
        "token_texts": token_texts,
        "total_steps": total_steps,
        "embeddings": embeddings_list,
        "positional_encodings": positional_encodings_list,
        "sparse_info": sparse_info,
        "moe_info": moe_info
    })
    
    return response


@app.post("/api/step", response_model=StepResponse)
async def get_step(request: StepRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = sessions[request.session_id]
    computation_steps = session["computation_steps"]
    
    if request.step < 0 or request.step >= len(computation_steps):
        raise HTTPException(
            status_code=400, 
            detail=f"步骤索引 {request.step} 超出范围 [0, {len(computation_steps)-1}]"
        )
    
    step_info = computation_steps[request.step]
    
    step_descriptions = {
        "layer_norm_1": "对输入进行Layer Normalization（注意力前）",
        "layer_norm_2": "对残差连接结果进行Layer Normalization（FFN前）",
        "q_projection": "计算Query矩阵：Q = X @ W_q",
        "k_projection": "计算Key矩阵：K = X @ W_k",
        "v_projection": "计算Value矩阵：V = X @ W_v",
        "attention_scores": "计算注意力分数：scores = (Q @ K^T) / sqrt(d_k)",
        "attention_weights": "对注意力分数应用Softmax",
        "attention_output": "用注意力权重加权Value：output = weights @ V",
        "attention_projection": "多头注意力输出投影：output = attention_out @ W_o",
        "residual_1": "残差连接：x = x + attention_output",
        "ffn_hidden": "前馈网络第一层：hidden = x @ W1 + b1",
        "ffn_relu": "ReLU激活函数",
        "ffn_output": "前馈网络第二层：output = hidden @ W2 + b2",
        "residual_2": "残差连接：x = x + ffn_output",
        "sparse_mask": "生成稀疏注意力掩码",
        "masked_attention": "应用稀疏掩码到注意力权重",
        "sparse_attention_output": "稀疏注意力输出",
        "moe_gate_logits": "计算MoE门控logits",
        "moe_gate_probs": "MoE门控概率",
        "expert_0_output": "专家0输出",
        "expert_1_output": "专家1输出",
        "expert_2_output": "专家2输出",
        "expert_3_output": "专家3输出",
        "expert_4_output": "专家4输出",
        "expert_5_output": "专家5输出",
        "expert_6_output": "专家6输出",
        "expert_7_output": "专家7输出",
        "moe_combined_output": "MoE专家组合输出"
    }
    
    # Get precision from session config
    precision = session["config"].viz_config.precision if session["config"].viz_config else 4
    
    response = StepResponse(
        step=request.step,
        step_type=step_info["step_type"],
        layer_index=step_info["layer_index"],
        description=step_descriptions.get(
            step_info["step_type"], 
            f"步骤: {step_info['step_type']}"
        ),
        input_data=numpy_to_list(step_info["input_data"], precision),
        output_data=numpy_to_list(step_info["output_data"], precision),
        metadata=step_info["metadata"],
        sparsity_info=step_info.get("sparsity_info"),
        moe_routing=step_info.get("moe_routing")
    )
    
    return response


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    del sessions[session_id]
    return {"message": "会话已删除", "session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    return {
        "total_sessions": len(sessions),
        "session_ids": list(sessions.keys())
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint with performance metrics"""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "cached_computations": len(computation_cache),
        "cache_hit_rate": "N/A"  # Could track this with counters
    }


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "cache_size": len(computation_cache),
        "max_cache_size": 100,
        "cached_keys": list(computation_cache.keys())[:10]  # First 10 keys
    }


@app.delete("/api/cache/clear")
async def clear_cache():
    """Clear all cached computations"""
    computation_cache.clear()
    return {
        "message": "Cache cleared successfully",
        "remaining_entries": len(computation_cache)
    }


@app.post("/api/trace", response_model=TraceResponse)
async def get_trace(request: TraceRequest):
    """获取指定输入的一次性全量计算trace（用于滚动叙事与离线演示）"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = sessions[request.session_id]
    computation_steps = session["computation_steps"]
    precision = request.precision or 4
    
    # 构建步骤响应
    step_descriptions = {
        "layer_norm_1": "对输入进行Layer Normalization（注意力前）",
        "layer_norm_2": "对残差连接结果进行Layer Normalization（FFN前）",
        "q_projection": "计算Query矩阵：Q = X @ W_q",
        "k_projection": "计算Key矩阵：K = X @ W_k",
        "v_projection": "计算Value矩阵：V = X @ W_v",
        "attention_scores": "计算注意力分数：scores = (Q @ K^T) / sqrt(d_k)",
        "attention_weights": "对注意力分数应用Softmax",
        "attention_output": "用注意力权重加权Value：output = weights @ V",
        "attention_projection": "多头注意力输出投影：output = attention_out @ W_o",
        "residual_1": "残差连接：x = x + attention_output",
        "ffn_hidden": "前馈网络第一层：hidden = x @ W1 + b1",
        "ffn_relu": "ReLU激活函数",
        "ffn_output": "前馈网络第二层：output = hidden @ W2 + b2",
        "residual_2": "残差连接：x = x + ffn_output",
        "sparse_mask": "生成稀疏注意力掩码",
        "masked_attention": "应用稀疏掩码到注意力权重",
        "sparse_attention_output": "稀疏注意力输出",
        "moe_gate_logits": "计算MoE门控logits",
        "moe_gate_probs": "MoE门控概率",
        "moe_combined_output": "MoE专家组合输出"
    }
    
    trace_steps = []
    start_time = time.time()
    
    for i, step_info in enumerate(computation_steps):
        metadata = step_info["metadata"] if request.include_metadata else {}
        
        step_response = StepResponse(
            step=i,
            step_type=step_info["step_type"],
            layer_index=step_info["layer_index"],
            description=step_descriptions.get(
                step_info["step_type"], 
                f"步骤: {step_info['step_type']}"
            ),
            input_data=numpy_to_list(step_info["input_data"], precision),
            output_data=numpy_to_list(step_info["output_data"], precision),
            metadata=metadata,
            sparsity_info=step_info.get("sparsity_info"),
            moe_routing=step_info.get("moe_routing")
        )
        trace_steps.append(step_response)
    
    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    
    # 获取最终输出
    final_output = computation_steps[-1]["output_data"] if computation_steps else np.array([])
    
    response = TraceResponse(
        session_id=request.session_id,
        input_text=" ".join(session["token_texts"]),
        tokens=session["token_ids"],
        token_texts=session["token_texts"],
        total_steps=len(computation_steps),
        computation_trace=trace_steps,
        final_output=numpy_to_list(final_output, precision),
        execution_time_ms=execution_time_ms
    )
    
    return response


@app.post("/api/chunk", response_model=ChunkedResponse)
async def get_chunk(request: ChunkRequest):
    """获取分块数据用于分页加载"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = sessions[request.session_id]
    computation_steps = session["computation_steps"]
    precision = session["config"].viz_config.precision if session["config"].viz_config else 4
    
    total_steps = len(computation_steps)
    chunk_size = request.chunk_size
    chunk_id = request.chunk_id
    
    # 计算分块范围
    start_step = chunk_id * chunk_size
    end_step = min(start_step + chunk_size, total_steps)
    
    if start_step >= total_steps:
        raise HTTPException(
            status_code=400, 
            detail=f"分块ID {chunk_id} 超出范围"
        )
    
    # 获取分块数据
    chunk_data = []
    for i in range(start_step, end_step):
        step_info = computation_steps[i]
        chunk_data.append({
            "step": i,
            "step_type": step_info["step_type"],
            "layer_index": step_info["layer_index"],
            "input_data": numpy_to_list(step_info["input_data"], precision),
            "output_data": numpy_to_list(step_info["output_data"], precision),
            "metadata": step_info["metadata"],
            "sparsity_info": step_info.get("sparsity_info"),
            "moe_routing": step_info.get("moe_routing")
        })
    
    total_chunks = (total_steps + chunk_size - 1) // chunk_size
    has_more = chunk_id + 1 < total_chunks
    
    response = ChunkedResponse(
        chunk_id=chunk_id,
        total_chunks=total_chunks,
        data=chunk_data,
        has_more=has_more
    )
    
    return response


async def stream_steps(session_id: str) -> AsyncGenerator[str, None]:
    """Stream computation steps as Server-Sent Events"""
    if session_id not in sessions:
        error_msg = {"type": "error", "data": {"message": "Session not found", "session_id": session_id}}
        yield f"data: {json.dumps(error_msg)}\n\n"
        return
    
    session = sessions[session_id]
    computation_steps = session["computation_steps"]
    precision = session["config"].viz_config.precision if session["config"].viz_config else 4
    
    step_descriptions = {
        "layer_norm_1": "对输入进行Layer Normalization（注意力前）",
        "layer_norm_2": "对残差连接结果进行Layer Normalization（FFN前）",
        "q_projection": "计算Query矩阵：Q = X @ W_q",
        "k_projection": "计算Key矩阵：K = X @ W_k",
        "v_projection": "计算Value矩阵：V = X @ W_v",
        "attention_scores": "计算注意力分数：scores = (Q @ K^T) / sqrt(d_k)",
        "attention_weights": "对注意力分数应用Softmax",
        "attention_output": "用注意力权重加权Value：output = weights @ V",
        "attention_projection": "多头注意力输出投影：output = attention_out @ W_o",
        "residual_1": "残差连接：x = x + attention_output",
        "ffn_hidden": "前馈网络第一层：hidden = x @ W1 + b1",
        "ffn_relu": "ReLU激活函数",
        "ffn_output": "前馈网络第二层：output = hidden @ W2 + b2",
        "residual_2": "残差连接：x = x + ffn_output",
        "sparse_mask": "生成稀疏注意力掩码",
        "masked_attention": "应用稀疏掩码到注意力权重",
        "sparse_attention_output": "稀疏注意力输出",
        "moe_gate_logits": "计算MoE门控logits",
        "moe_gate_probs": "MoE门控概率",
        "moe_combined_output": "MoE专家组合输出"
    }
    
    # Send start message
    start_msg = {"type": "start", "data": {"total_steps": len(computation_steps), "session_id": session_id}}
    yield f"data: {json.dumps(start_msg)}\n\n"
    
    try:
        for i, step_info in enumerate(computation_steps):
            step_response = {
                "type": "step",
                "data": {
                    "step": i,
                    "step_type": step_info["step_type"],
                    "layer_index": step_info["layer_index"],
                    "description": step_descriptions.get(
                        step_info["step_type"], 
                        f"步骤: {step_info['step_type']}"
                    ),
                    "input_data": numpy_to_list(step_info["input_data"], precision),
                    "output_data": numpy_to_list(step_info["output_data"], precision),
                    "metadata": step_info["metadata"],
                    "sparsity_info": step_info.get("sparsity_info"),
                    "moe_routing": step_info.get("moe_routing"),
                    "timestamp": time.time()
                }
            }
            
            # Send as SSE
            yield f"data: {json.dumps(step_response)}\n\n"
            
            # Small delay to simulate processing (configurable)
            delay = 0.05 if session["config"].viz_config and session["config"].viz_config.animate_transitions else 0.01
            await asyncio.sleep(delay)
        
        # Send completion message
        complete_msg = {"type": "complete", "data": {"message": "Streaming completed", "total_steps": len(computation_steps)}}
        yield f"data: {json.dumps(complete_msg)}\n\n"
        
    except Exception as e:
        error_msg = {"type": "error", "data": {"message": f"Streaming error: {str(e)}", "step": i}}
        yield f"data: {json.dumps(error_msg)}\n\n"


@app.get("/api/stream/{session_id}")
async def stream_computation(session_id: str):
    """Stream computation steps as Server-Sent Events"""
    return StreamingResponse(
        stream_steps(session_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time step streaming"""
    await websocket.accept()
    
    if session_id not in sessions:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Session not found", "session_id": session_id}
        })
        await websocket.close()
        return
    
    # Add to active connections
    connection_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
    websocket_connections[connection_id] = websocket
    
    try:
        session = sessions[session_id]
        computation_steps = session["computation_steps"]
        precision = session["config"].viz_config.precision if session["config"].viz_config else 4
        
        # Send initial info
        await websocket.send_json({
            "type": "connected",
            "data": {
                "session_id": session_id,
                "connection_id": connection_id,
                "total_steps": len(computation_steps)
            }
        })
        
        # Listen for client messages
        while True:
            try:
                message = await websocket.receive_json()
                message_type = message.get("type")
                
                if message_type == "subscribe":
                    # Start streaming all steps
                    await stream_steps_websocket(websocket, session_id, computation_steps, precision)
                
                elif message_type == "step":
                    # Send specific step
                    step_index = message.get("data", {}).get("step")
                    if step_index is not None and 0 <= step_index < len(computation_steps):
                        await send_step_websocket(websocket, computation_steps[step_index], step_index, precision)
                
                elif message_type == "trace":
                    # Send full trace
                    await send_trace_websocket(websocket, session, computation_steps, precision)
                
                elif message_type == "ping":
                    # Respond to ping
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": f"Unknown message type: {message_type}"}
                    })
                    
            except Exception as e:
                await websocket.send_json({
                    "type": "error", 
                    "data": {"message": f"Message processing error: {str(e)}"}
                })
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        # Remove from active connections
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]


async def stream_steps_websocket(websocket: WebSocket, session_id: str, computation_steps: List, precision: int):
    """Stream steps via WebSocket"""
    step_descriptions = {
        "layer_norm_1": "对输入进行Layer Normalization（注意力前）",
        "layer_norm_2": "对残差连接结果进行Layer Normalization（FFN前）",
        "q_projection": "计算Query矩阵：Q = X @ W_q",
        "k_projection": "计算Key矩阵：K = X @ W_k",
        "v_projection": "计算Value矩阵：V = X @ W_v",
        "attention_scores": "计算注意力分数：scores = (Q @ K^T) / sqrt(d_k)",
        "attention_weights": "对注意力分数应用Softmax",
        "attention_output": "用注意力权重加权Value：output = weights @ V",
        "attention_projection": "多头注意力输出投影：output = attention_out @ W_o",
        "residual_1": "残差连接：x = x + attention_output",
        "ffn_hidden": "前馈网络第一层：hidden = x @ W1 + b1",
        "ffn_relu": "ReLU激活函数",
        "ffn_output": "前馈网络第二层：output = hidden @ W2 + b2",
        "residual_2": "残差连接：x = x + ffn_output",
    }
    
    for i, step_info in enumerate(computation_steps):
        await websocket.send_json({
            "type": "step",
            "data": {
                "step": i,
                "step_type": step_info["step_type"],
                "layer_index": step_info["layer_index"],
                "description": step_descriptions.get(step_info["step_type"], f"步骤: {step_info['step_type']}"),
                "input_data": numpy_to_list(step_info["input_data"], precision),
                "output_data": numpy_to_list(step_info["output_data"], precision),
                "metadata": step_info["metadata"],
                "sparsity_info": step_info.get("sparsity_info"),
                "moe_routing": step_info.get("moe_routing"),
                "timestamp": time.time()
            }
        })
        
        # Small delay
        await asyncio.sleep(0.01)
    
    await websocket.send_json({
        "type": "complete",
        "data": {"message": "Streaming completed", "total_steps": len(computation_steps)}
    })


async def send_step_websocket(websocket: WebSocket, step_info: Dict, step_index: int, precision: int):
    """Send single step via WebSocket"""
    step_descriptions = {
        "layer_norm_1": "对输入进行Layer Normalization（注意力前）",
        "layer_norm_2": "对残差连接结果进行Layer Normalization（FFN前）",
        "q_projection": "计算Query矩阵：Q = X @ W_q",
        "k_projection": "计算Key矩阵：K = X @ W_k",
        "v_projection": "计算Value矩阵：V = X @ W_v",
        "attention_scores": "计算注意力分数：scores = (Q @ K^T) / sqrt(d_k)",
        "attention_weights": "对注意力分数应用Softmax",
        "attention_output": "用注意力权重加权Value：output = weights @ V",
        "attention_projection": "多头注意力输出投影：output = attention_out @ W_o",
        "residual_1": "残差连接：x = x + attention_output",
        "ffn_hidden": "前馈网络第一层：hidden = x @ W1 + b1",
        "ffn_relu": "ReLU激活函数",
        "ffn_output": "前馈网络第二层：output = hidden @ W2 + b2",
        "residual_2": "残差连接：x = x + ffn_output",
    }
    
    await websocket.send_json({
        "type": "step",
        "data": {
            "step": step_index,
            "step_type": step_info["step_type"],
            "layer_index": step_info["layer_index"],
            "description": step_descriptions.get(step_info["step_type"], f"步骤: {step_info['step_type']}"),
            "input_data": numpy_to_list(step_info["input_data"], precision),
            "output_data": numpy_to_list(step_info["output_data"], precision),
            "metadata": step_info["metadata"],
            "sparsity_info": step_info.get("sparsity_info"),
            "moe_routing": step_info.get("moe_routing"),
            "timestamp": time.time()
        }
    })


async def send_trace_websocket(websocket: WebSocket, session: Dict, computation_steps: List, precision: int):
    """Send full trace via WebSocket"""
    trace_data = {
        "session_id": session["config"],
        "tokens": session["token_ids"],
        "token_texts": session["token_texts"],
        "total_steps": len(computation_steps),
        "computation_trace": []
    }
    
    step_descriptions = {
        "layer_norm_1": "对输入进行Layer Normalization（注意力前）",
        "layer_norm_2": "对残差连接结果进行Layer Normalization（FFN前）",
        "q_projection": "计算Query矩阵：Q = X @ W_q",
        "k_projection": "计算Key矩阵：K = X @ W_k",
        "v_projection": "计算Value矩阵：V = X @ W_v",
        "attention_scores": "计算注意力分数：scores = (Q @ K^T) / sqrt(d_k)",
        "attention_weights": "对注意力分数应用Softmax",
        "attention_output": "用注意力权重加权Value：output = weights @ V",
        "attention_projection": "多头注意力输出投影：output = attention_out @ W_o",
        "residual_1": "残差连接：x = x + attention_output",
        "ffn_hidden": "前馈网络第一层：hidden = x @ W1 + b1",
        "ffn_relu": "ReLU激活函数",
        "ffn_output": "前馈网络第二层：output = hidden @ W2 + b2",
        "residual_2": "残差连接：x = x + ffn_output",
    }
    
    for i, step_info in enumerate(computation_steps):
        step_data = {
            "step": i,
            "step_type": step_info["step_type"],
            "layer_index": step_info["layer_index"],
            "description": step_descriptions.get(step_info["step_type"], f"步骤: {step_info['step_type']}"),
            "input_data": numpy_to_list(step_info["input_data"], precision),
            "output_data": numpy_to_list(step_info["output_data"], precision),
            "metadata": step_info["metadata"],
            "sparsity_info": step_info.get("sparsity_info"),
            "moe_routing": step_info.get("moe_routing")
        }
        trace_data["computation_trace"].append(step_data)
    
    await websocket.send_json({
        "type": "trace",
        "data": trace_data
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
