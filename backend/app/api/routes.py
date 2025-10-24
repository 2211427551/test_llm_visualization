from fastapi import APIRouter, HTTPException
from app.models import ModelForwardRequest, ModelForwardResponse, Token, ComputationStep, LayerData, OutputProbability
import numpy as np

router = APIRouter()


def tokenize(text: str) -> list[Token]:
    words = text.strip().split()
    tokens = [Token(text=word, id=idx) for idx, word in enumerate(words)]
    return tokens


def simulate_layer_computation(layer_id: int, layer_name: str, input_shape: list[int], output_shape: list[int], truncate: bool = False):
    activations = np.random.randn(min(input_shape[0], 8), min(input_shape[1] if len(input_shape) > 1 else 10, 10)).tolist() if not truncate else None
    weights = np.random.randn(min(output_shape[0], 8), min(output_shape[1] if len(output_shape) > 1 else 10, 10)).tolist() if layer_id % 2 == 0 else None
    
    return LayerData(
        layerId=layer_id,
        layerName=layer_name,
        inputShape=input_shape,
        outputShape=output_shape,
        activations=activations,
        weights=weights,
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
                [token_count, 1], 
                [token_count, embedding_dim],
                truncate=truncated
            ),
            description="Convert tokens to dense vector representations"
        ),
        ComputationStep(
            stepIndex=1,
            layerData=simulate_layer_computation(
                1, 
                "Multi-Head Attention", 
                [token_count, embedding_dim], 
                [token_count, embedding_dim],
                truncate=truncated
            ),
            description="Apply self-attention mechanism across tokens"
        ),
        ComputationStep(
            stepIndex=2,
            layerData=simulate_layer_computation(
                2, 
                "Feed Forward Network", 
                [token_count, embedding_dim], 
                [token_count, hidden_dim],
                truncate=truncated
            ),
            description="Transform through position-wise feed-forward layers"
        ),
        ComputationStep(
            stepIndex=3,
            layerData=simulate_layer_computation(
                3, 
                "Layer Normalization", 
                [token_count, hidden_dim], 
                [token_count, hidden_dim],
                truncate=truncated
            ),
            description="Normalize activations for training stability"
        ),
        ComputationStep(
            stepIndex=4,
            layerData=simulate_layer_computation(
                4, 
                "Output Projection", 
                [token_count, hidden_dim], 
                [token_count, output_dim],
                truncate=truncated
            ),
            description="Project to output vocabulary space"
        ),
        ComputationStep(
            stepIndex=5,
            layerData=simulate_layer_computation(
                5, 
                "Softmax Layer", 
                [token_count, output_dim], 
                [token_count, output_dim],
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
