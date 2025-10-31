from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import json
import asyncio
from app.models.request import InitRequest, StepRequest
from app.models.response import InitResponse, StepResponse, InitialState
from app.services.tokenizer import SimpleTokenizer
from app.services.embedding import EmbeddingLayer
from app.services.transformer import TransformerSimulator
import uuid
from typing import Dict, Any, AsyncGenerator
import numpy as np
from functools import lru_cache
import hashlib

# Add streaming support to existing main.py
sessions: Dict[str, Dict[str, Any]] = {}
computation_cache: Dict[str, Any] = {}

def numpy_to_list(arr: np.ndarray) -> list:
    """Convert numpy array to list with reduced precision for smaller JSON size"""
    if isinstance(arr, np.ndarray):
        # Round to 4 decimal places to reduce data size
        return np.round(arr, 4).tolist()
    return arr

async def stream_steps(session_id: str) -> AsyncGenerator[str, None]:
    """Stream computation steps as Server-Sent Events"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    computation_steps = session["computation_steps"]
    
    for i, step_info in enumerate(computation_steps):
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
        
        step_response = {
            "step": i,
            "step_type": step_info["step_type"],
            "layer_index": step_info["layer_index"],
            "description": step_descriptions.get(
                step_info["step_type"], 
                f"步骤: {step_info['step_type']}"
            ),
            "input_data": numpy_to_list(step_info["input_data"]),
            "output_data": numpy_to_list(step_info["output_data"]),
            "metadata": step_info["metadata"]
        }
        
        # Send as SSE
        yield f"data: {json.dumps(step_response)}\n\n"
        
        # Small delay to simulate processing
        await asyncio.sleep(0.1)

def add_streaming_endpoints(app: FastAPI):
    """Add streaming endpoints to existing FastAPI app"""
    
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
    
    @app.post("/api/init-extended", response_model=InitResponse)
    async def initialize_session_extended(request: InitRequest):
        """Extended init with additional metadata"""
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