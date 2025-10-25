from fastapi import APIRouter, HTTPException, Query
from app.models import (
    ModelForwardRequest, ModelForwardResponse, Token, ComputationStep, 
    LayerData, OutputProbability, AttentionData, MoEData, ExpertData,
    ModelRunResponse, ForwardStep, LayerState, TracedTensor, TensorMetadata,
    AttentionState, MoEState, ExpertActivation, ModelMetadata
)
from app.model import SimpleTransformerModel, LayerTrace, TensorTrace
from app.tokenizer import get_tokenizer
import numpy as np
import time
from typing import Optional

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


def tensor_trace_to_pydantic(trace: TensorTrace) -> TracedTensor:
    """Convert TensorTrace to Pydantic TracedTensor."""
    # Convert numpy array to nested list
    if isinstance(trace.data, np.ndarray):
        data = trace.data.tolist()
    else:
        data = trace.data
    
    metadata = TensorMetadata(
        shape=trace.shape,
        dtype=trace.dtype,
        truncated=trace.truncated,
        minVal=trace.min_val,
        maxVal=trace.max_val,
        meanVal=trace.mean_val,
        stdVal=trace.std_val
    )
    
    return TracedTensor(
        name=trace.name,
        data=data,
        metadata=metadata
    )


def layer_trace_to_pydantic(trace: LayerTrace, step_index: int, timing_ms: Optional[float] = None) -> ForwardStep:
    """Convert LayerTrace to Pydantic ForwardStep."""
    # Determine input/output shapes
    input_shape = trace.pre_activations.shape if trace.pre_activations else [0]
    output_shape = trace.post_activations.shape if trace.post_activations else [0]
    
    # Build LayerState
    layer_state_dict = {
        "layerId": trace.layer_id,
        "layerName": trace.layer_name,
        "layerType": trace.layer_type,
        "inputShape": input_shape,
        "outputShape": output_shape,
        "metadata": trace.metadata
    }
    
    # Add pre/post activations
    if trace.pre_activations:
        layer_state_dict["preActivations"] = tensor_trace_to_pydantic(trace.pre_activations)
    if trace.post_activations:
        layer_state_dict["postActivations"] = tensor_trace_to_pydantic(trace.post_activations)
    if trace.weights:
        layer_state_dict["weights"] = tensor_trace_to_pydantic(trace.weights)
    
    # Add attention state if present
    if trace.attention_data:
        attention_state = AttentionState(
            queryMatrix=tensor_trace_to_pydantic(trace.attention_data["query_matrix"]),
            keyMatrix=tensor_trace_to_pydantic(trace.attention_data["key_matrix"]),
            valueMatrix=tensor_trace_to_pydantic(trace.attention_data["value_matrix"]),
            attentionScores=tensor_trace_to_pydantic(trace.attention_data["attention_scores"]),
            sparsityMask=tensor_trace_to_pydantic(trace.attention_data["sparsity_mask"]) if trace.attention_data.get("sparsity_mask") else None,
            numHeads=trace.attention_data["num_heads"],
            headDim=trace.attention_data["head_dim"]
        )
        layer_state_dict["attentionState"] = attention_state
    
    # Add MoE state if present
    if trace.moe_data:
        expert_activations = [
            ExpertActivation(
                expertId=exp["expert_id"],
                activations=tensor_trace_to_pydantic(exp["activations"])
            )
            for exp in trace.moe_data["expert_activations"]
        ]
        moe_state = MoEState(
            gatingWeights=tensor_trace_to_pydantic(trace.moe_data["gating_weights"]),
            selectedExperts=trace.moe_data["selected_experts"],
            expertActivations=expert_activations,
            numExperts=trace.moe_data["num_experts"],
            topK=trace.moe_data["top_k"]
        )
        layer_state_dict["moeState"] = moe_state
    
    layer_state = LayerState(**layer_state_dict)
    
    # Build description
    descriptions = {
        "embedding": "Convert tokens to dense vector representations",
        "attention": "Apply multi-head self-attention mechanism across tokens",
        "moe": "Route tokens to specialized expert networks using top-k gating",
        "normalization": "Normalize activations for training stability",
        "feedforward": "Project to output vocabulary space",
        "output": "Compute probability distribution over tokens"
    }
    description = descriptions.get(trace.layer_type, f"Process through {trace.layer_name}")
    
    return ForwardStep(
        stepIndex=step_index,
        layerState=layer_state,
        description=description,
        timingMs=timing_ms
    )


@router.post("/model/forward/traced", response_model=ModelRunResponse)
async def model_forward_traced(
    request: ModelForwardRequest,
    tokenizer_type: str = Query("bpe", description="Tokenizer type: 'char' or 'bpe'"),
    step_index: Optional[int] = Query(None, description="Return only specific step (0-indexed)")
):
    """
    Run model forward pass with complete tracing of all intermediate tensors.
    
    This endpoint implements the forward tracing requirements:
    - Accumulates all intermediate tensors (embeddings, Q/K/V, attention, MoE, logits)
    - Returns structured ForwardStep objects with LayerState hierarchy
    - Includes timing metadata
    - Supports step slicing via step_index query parameter
    - Automatically truncates large tensors with min/max summaries
    
    Args:
        request: ModelForwardRequest with text input
        tokenizer_type: Type of tokenizer ("char" or "bpe")
        step_index: Optional step index to return only that step
        
    Returns:
        ModelRunResponse with complete forward pass trace
    """
    text = request.text.strip()
    
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Input text exceeds maximum length of 500 characters")
    
    start_time = time.time()
    warnings = []
    
    # Initialize tokenizer
    try:
        tokenizer = get_tokenizer(tokenizer_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tokenizer type: {tokenizer_type}")
    
    # Tokenize input
    token_ids, token_info = tokenizer.encode(text)
    token_count = len(token_ids)
    
    if token_count == 0:
        warnings.append("Input text produced no tokens after tokenization")
    
    if token_count > 16:
        warnings.append(f"Token count ({token_count}) exceeds recommended limit of 16. Some tensors will be truncated.")
    
    # Create tokens list
    tokens = [
        Token(
            text=info.text,
            id=info.id,
            startPos=info.start_pos,
            endPos=info.end_pos
        )
        for info in token_info
    ]
    
    # Initialize model
    model = SimpleTransformerModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_heads=4,
        num_experts=8,
        top_k_experts=2
    )
    
    # Run forward pass with tracing
    layer_traces = model.forward(token_ids)
    
    total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Convert traces to ForwardStep objects
    steps = []
    for i, trace in enumerate(layer_traces):
        # Estimate timing per layer (distributed)
        layer_timing = total_time / len(layer_traces)
        forward_step = layer_trace_to_pydantic(trace, i, layer_timing)
        steps.append(forward_step)
    
    # Filter by step_index if requested
    if step_index is not None:
        if step_index < 0 or step_index >= len(steps):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid step_index {step_index}. Valid range: 0-{len(steps)-1}"
            )
        steps = [steps[step_index]]
    
    # Get final logits and probabilities
    final_trace = layer_traces[-1]
    final_logits_tensor = final_trace.post_activations
    final_logits = tensor_trace_to_pydantic(final_logits_tensor)
    
    # Compute output probabilities (average across tokens)
    probabilities_array = final_logits_tensor.data
    if len(probabilities_array.shape) == 2:
        avg_probs = np.mean(probabilities_array, axis=0)
    else:
        avg_probs = probabilities_array
    
    # Get top-5 predictions
    top_k = min(5, len(avg_probs))
    top_indices = np.argsort(avg_probs)[-top_k:][::-1]
    
    output_probs = []
    for idx in top_indices:
        # Try to decode token
        if hasattr(tokenizer, 'id_to_word') and idx < len(tokenizer.id_to_word):
            token_text = tokenizer.id_to_word.get(int(idx), f"token_{idx}")
        elif hasattr(tokenizer, 'id_to_char') and idx < len(tokenizer.id_to_char):
            token_text = tokenizer.id_to_char.get(int(idx), f"char_{idx}")
        else:
            token_text = f"token_{idx}"
        
        output_probs.append(OutputProbability(
            token=token_text,
            probability=float(avg_probs[idx])
        ))
    
    # Build metadata
    metadata = ModelMetadata(
        vocabSize=model.vocab_size,
        embeddingDim=model.embedding_dim,
        hiddenDim=model.hidden_dim,
        numLayers=len(layer_traces),
        tokenizerType=tokenizer_type,
        totalTimeMs=total_time
    )
    
    return ModelRunResponse(
        success=True,
        inputText=text,
        tokens=tokens,
        tokenCount=token_count,
        steps=steps,
        finalLogits=final_logits,
        outputProbabilities=output_probs,
        metadata=metadata,
        warnings=warnings if warnings else None
    )


@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}
