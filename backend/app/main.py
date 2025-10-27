from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.request import InitRequest, StepRequest
from app.models.response import InitResponse, StepResponse, InitialState
from app.services.tokenizer import SimpleTokenizer
from app.services.embedding import EmbeddingLayer
from app.services.transformer import TransformerSimulator
import uuid
from typing import Dict, Any
import numpy as np


app = FastAPI(
    title="Transformer计算模拟器API",
    description="标准Transformer逐步计算可视化后端",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, Dict[str, Any]] = {}


def numpy_to_list(arr: np.ndarray) -> list:
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


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
    
    sessions[session_id] = {
        "config": request.config,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "embeddings": embeddings,
        "positional_encodings": positional_encodings,
        "initial_input": initial_input,
        "computation_steps": computation_steps,
        "tokenizer": tokenizer
    }
    
    total_steps = len(computation_steps)
    
    response = InitResponse(
        session_id=session_id,
        tokens=token_ids,
        token_texts=token_texts,
        total_steps=total_steps,
        initial_state=InitialState(
            embeddings=numpy_to_list(embeddings),
            positional_encodings=numpy_to_list(positional_encodings)
        )
    )
    
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
