from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.models.request import InitRequest, StepRequest
from app.models.response import InitResponse, StepResponse, InitialState
from app.services.tokenizer import SimpleTokenizer
from app.services.embedding import EmbeddingLayer
from app.services.transformer import TransformerSimulator
import uuid
from typing import Dict, Any
import numpy as np
from functools import lru_cache
import hashlib
import json


app = FastAPI(
    title="Transformer计算模拟器API",
    description="标准Transformer逐步计算可视化后端",
    version="1.0.0"
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


def numpy_to_list(arr: np.ndarray) -> list:
    """Convert numpy array to list with reduced precision for smaller JSON size"""
    if isinstance(arr, np.ndarray):
        # Round to 4 decimal places to reduce data size
        return np.round(arr, 4).tolist()
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
            )
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
    
    transformer = TransformerSimulator(
        request.config.n_embd,
        request.config.n_layer,
        request.config.n_head
    )
    
    computation_steps = transformer.simulate(initial_input)
    
    session_data = {
        "config": request.config,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "embeddings": embeddings,
        "positional_encodings": positional_encodings,
        "initial_input": initial_input,
        "computation_steps": computation_steps,
        "tokenizer": tokenizer
    }
    
    sessions[session_id] = session_data
    
    total_steps = len(computation_steps)
    
    # Prepare response
    embeddings_list = numpy_to_list(embeddings)
    positional_encodings_list = numpy_to_list(positional_encodings)
    
    response = InitResponse(
        session_id=session_id,
        tokens=token_ids,
        token_texts=token_texts,
        total_steps=total_steps,
        initial_state=InitialState(
            embeddings=embeddings_list,
            positional_encodings=positional_encodings_list
        )
    )
    
    # Cache the result
    cache_computation(cache_key, {
        "session_data": session_data,
        "tokens": token_ids,
        "token_texts": token_texts,
        "total_steps": total_steps,
        "embeddings": embeddings_list,
        "positional_encodings": positional_encodings_list
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
        "residual_2": "残差连接：x = x + ffn_output"
    }
    
    response = StepResponse(
        step=request.step,
        step_type=step_info["step_type"],
        layer_index=step_info["layer_index"],
        description=step_descriptions.get(
            step_info["step_type"], 
            f"步骤: {step_info['step_type']}"
        ),
        input_data=numpy_to_list(step_info["input_data"]),
        output_data=numpy_to_list(step_info["output_data"]),
        metadata=step_info["metadata"]
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
