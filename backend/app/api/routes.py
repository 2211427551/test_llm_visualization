from fastapi import APIRouter, HTTPException
from app.models import (
    ModelForwardRequest, ModelForwardResponse, Token, ComputationStep, 
    LayerData, OutputProbability, AttentionData, MoEData, ExpertData
)
import numpy as np

router = APIRouter()


def tokenize(text: str) -> list[Token]:
    words = text.strip().split()
    tokens = [Token(text=word, id=idx) for idx, word in enumerate(words)]
    return tokens


def simulate_attention_data(num_tokens: int, num_heads: int = 4, head_dim: int = 16, truncate: bool = False):
    """Generate sample attention mechanism data"""
    if truncate:
        return None
    
    # Limit to max 16 tokens for performance
    num_tokens = min(num_tokens, 16)
    
    # Q, K, V matrices: [num_tokens, num_heads * head_dim]
    embedding_dim = num_heads * head_dim
    q_matrix = np.random.randn(num_tokens, embedding_dim) * 0.5
    k_matrix = np.random.randn(num_tokens, embedding_dim) * 0.5
    v_matrix = np.random.randn(num_tokens, embedding_dim) * 0.5
    
    # Attention scores: [num_tokens, num_tokens] with softmax-like distribution
    raw_scores = np.random.randn(num_tokens, num_tokens)
    attention_scores = np.exp(raw_scores) / np.exp(raw_scores).sum(axis=1, keepdims=True)
    
    # Sparsity mask: randomly prune ~30% of attention weights
    sparsity_mask = (np.random.rand(num_tokens, num_tokens) > 0.3).astype(int)
    # Apply mask to attention scores
    attention_scores = attention_scores * sparsity_mask
    # Re-normalize
    row_sums = attention_scores.sum(axis=1, keepdims=True)
    attention_scores = np.where(row_sums > 0, attention_scores / row_sums, 0)
    
    return AttentionData(
        queryMatrix=q_matrix.tolist(),
        keyMatrix=k_matrix.tolist(),
        valueMatrix=v_matrix.tolist(),
        attentionScores=attention_scores.tolist(),
        sparsityMask=sparsity_mask.tolist(),
        numHeads=num_heads,
        headDim=head_dim
    )


def simulate_moe_data(num_tokens: int, num_experts: int = 8, top_k: int = 2, hidden_dim: int = 64, truncate: bool = False):
    """Generate sample Mixture of Experts data"""
    if truncate:
        return None
    
    # Limit to max 16 tokens for performance
    num_tokens = min(num_tokens, 16)
    
    # Gating weights: [num_tokens, num_experts] with softmax distribution
    raw_gates = np.random.randn(num_tokens, num_experts)
    gating_weights = np.exp(raw_gates) / np.exp(raw_gates).sum(axis=1, keepdims=True)
    
    # Selected experts: top-k experts per token
    selected_experts = np.argsort(gating_weights, axis=1)[:, -top_k:].tolist()
    
    # Expert activations: histogram data for each expert
    expert_activations = []
    for expert_id in range(num_experts):
        # Generate activation values with some variety
        activations = np.random.randn(hidden_dim) * (0.5 + expert_id * 0.1)
        expert_activations.append(ExpertData(
            expertId=expert_id,
            activations=activations.tolist()
        ))
    
    return MoEData(
        gatingWeights=gating_weights.tolist(),
        selectedExperts=selected_experts,
        expertActivations=expert_activations,
        numExperts=num_experts,
        topK=top_k
    )


def simulate_layer_computation(
    layer_id: int, 
    layer_name: str, 
    layer_type: str,
    input_shape: list[int], 
    output_shape: list[int], 
    num_tokens: int,
    truncate: bool = False
):
    activations = np.random.randn(min(input_shape[0], 8), min(input_shape[1] if len(input_shape) > 1 else 10, 10)).tolist() if not truncate else None
    weights = np.random.randn(min(output_shape[0], 8), min(output_shape[1] if len(output_shape) > 1 else 10, 10)).tolist() if layer_id % 2 == 0 else None
    
    # Add attention data for attention layers
    attention_data = None
    if layer_type == "attention":
        attention_data = simulate_attention_data(num_tokens, truncate=truncate)
    
    # Add MoE data for MoE layers
    moe_data = None
    if layer_type == "moe":
        moe_data = simulate_moe_data(num_tokens, truncate=truncate)
    
    return LayerData(
        layerId=layer_id,
        layerName=layer_name,
        layerType=layer_type,
        inputShape=input_shape,
        outputShape=output_shape,
        activations=activations,
        weights=weights,
        attentionData=attention_data,
        moeData=moe_data,
        truncated=truncate
    )


@router.post("/model/forward", response_model=ModelForwardResponse)
async def model_forward(request: ModelForwardRequest):
    text = request.text.strip()
    
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Input text exceeds maximum length of 500 characters")
    
    tokens = tokenize(text)
    token_count = len(tokens)
    
    warnings = []
    truncated = False
    
    if token_count > 100:
        warnings.append("Token count exceeds 100. Some tensor data has been truncated to reduce payload size.")
        truncated = True
    
    if len(text) > 400:
        warnings.append("Input text is very long. Backend is processing with limited tensor representation.")
    
    embedding_dim = 64
    hidden_dim = 128
    output_dim = 32
    
    steps = [
        ComputationStep(
            stepIndex=0,
            layerData=simulate_layer_computation(
                0, 
                "Embedding Layer",
                "embedding",
                [token_count, 1], 
                [token_count, embedding_dim],
                token_count,
                truncate=truncated
            ),
            description="Convert tokens to dense vector representations"
        ),
        ComputationStep(
            stepIndex=1,
            layerData=simulate_layer_computation(
                1, 
                "Multi-Head Attention",
                "attention",
                [token_count, embedding_dim], 
                [token_count, embedding_dim],
                token_count,
                truncate=truncated
            ),
            description="Apply self-attention mechanism across tokens"
        ),
        ComputationStep(
            stepIndex=2,
            layerData=simulate_layer_computation(
                2, 
                "Mixture of Experts (MoE)",
                "moe",
                [token_count, embedding_dim], 
                [token_count, hidden_dim],
                token_count,
                truncate=truncated
            ),
            description="Route tokens to specialized expert networks"
        ),
        ComputationStep(
            stepIndex=3,
            layerData=simulate_layer_computation(
                3, 
                "Layer Normalization",
                "normalization",
                [token_count, hidden_dim], 
                [token_count, hidden_dim],
                token_count,
                truncate=truncated
            ),
            description="Normalize activations for training stability"
        ),
        ComputationStep(
            stepIndex=4,
            layerData=simulate_layer_computation(
                4, 
                "Output Projection",
                "feedforward",
                [token_count, hidden_dim], 
                [token_count, output_dim],
                token_count,
                truncate=truncated
            ),
            description="Project to output vocabulary space"
        ),
        ComputationStep(
            stepIndex=5,
            layerData=simulate_layer_computation(
                5, 
                "Softmax Layer",
                "output",
                [token_count, output_dim], 
                [token_count, output_dim],
                token_count,
                truncate=truncated
            ),
            description="Compute probability distribution over tokens"
        ),
    ]
    
    vocab_sample = ["the", "is", "and", "to", "a", "of", "in", "that", "for", "it"]
    probabilities = np.random.dirichlet(np.ones(10))
    output_probs = [
        OutputProbability(token=vocab_sample[i], probability=float(probabilities[i]))
        for i in range(10)
    ]
    output_probs.sort(key=lambda x: x.probability, reverse=True)
    
    return ModelForwardResponse(
        success=True,
        inputText=text,
        tokens=tokens,
        tokenCount=token_count,
        steps=steps,
        outputProbabilities=output_probs[:5],
        warnings=warnings if warnings else None,
        truncated=truncated
    )


@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}
