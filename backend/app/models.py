from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class Token(BaseModel):
    text: str
    id: int
    startPos: Optional[int] = Field(None, alias="startPos")
    endPos: Optional[int] = Field(None, alias="endPos")
    
    class Config:
        populate_by_name = True


class AttentionData(BaseModel):
    queryMatrix: Optional[List[List[float]]] = Field(None, alias="queryMatrix")
    keyMatrix: Optional[List[List[float]]] = Field(None, alias="keyMatrix")
    valueMatrix: Optional[List[List[float]]] = Field(None, alias="valueMatrix")
    attentionScores: Optional[List[List[float]]] = Field(None, alias="attentionScores")
    sparsityMask: Optional[List[List[int]]] = Field(None, alias="sparsityMask")
    numHeads: Optional[int] = Field(None, alias="numHeads")
    headDim: Optional[int] = Field(None, alias="headDim")

    class Config:
        populate_by_name = True


class ExpertData(BaseModel):
    expertId: int = Field(alias="expertId")
    activations: List[float]
    
    class Config:
        populate_by_name = True


class MoEData(BaseModel):
    gatingWeights: Optional[List[List[float]]] = Field(None, alias="gatingWeights")
    selectedExperts: Optional[List[List[int]]] = Field(None, alias="selectedExperts")
    expertActivations: Optional[List[ExpertData]] = Field(None, alias="expertActivations")
    numExperts: Optional[int] = Field(None, alias="numExperts")
    topK: Optional[int] = Field(None, alias="topK")

    class Config:
        populate_by_name = True


class LayerData(BaseModel):
    layerId: int = Field(alias="layerId")
    layerName: str = Field(alias="layerName")
    layerType: Optional[Literal["attention", "moe", "feedforward", "embedding", "normalization", "output"]] = Field(None, alias="layerType")
    inputShape: List[int] = Field(alias="inputShape")
    outputShape: List[int] = Field(alias="outputShape")
    activations: Optional[List[List[float]]] = None
    weights: Optional[List[List[float]]] = None
    attentionData: Optional[AttentionData] = Field(None, alias="attentionData")
    moeData: Optional[MoEData] = Field(None, alias="moeData")
    truncated: Optional[bool] = False

    class Config:
        populate_by_name = True


class ComputationStep(BaseModel):
    stepIndex: int = Field(alias="stepIndex")
    layerData: LayerData = Field(alias="layerData")
    description: str


class OutputProbability(BaseModel):
    token: str
    probability: float


class ModelForwardRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


class ModelForwardResponse(BaseModel):
    success: bool
    inputText: str = Field(alias="inputText")
    tokens: List[Token]
    tokenCount: int = Field(alias="tokenCount")
    steps: List[ComputationStep]
    outputProbabilities: List[OutputProbability] = Field(alias="outputProbabilities")
    warnings: Optional[List[str]] = None
    truncated: Optional[bool] = False

    class Config:
        populate_by_name = True


# New schemas for forward tracing (as per ticket requirements)

class TensorMetadata(BaseModel):
    """Metadata for a traced tensor including statistics."""
    shape: List[int]
    dtype: str = "float32"
    truncated: bool = False
    minVal: float = Field(0.0, alias="minVal")
    maxVal: float = Field(0.0, alias="maxVal")
    meanVal: float = Field(0.0, alias="meanVal")
    stdVal: float = Field(0.0, alias="stdVal")
    
    class Config:
        populate_by_name = True


class TracedTensor(BaseModel):
    """A tensor with its data and metadata."""
    name: str
    data: List[Any]  # Can be List[float] or List[List[float]], etc.
    metadata: TensorMetadata
    
    class Config:
        populate_by_name = True


class AttentionState(BaseModel):
    """Detailed attention mechanism state."""
    queryMatrix: TracedTensor = Field(alias="queryMatrix")
    keyMatrix: TracedTensor = Field(alias="keyMatrix")
    valueMatrix: TracedTensor = Field(alias="valueMatrix")
    attentionScores: TracedTensor = Field(alias="attentionScores")
    sparsityMask: Optional[TracedTensor] = Field(None, alias="sparsityMask")
    numHeads: int = Field(alias="numHeads")
    headDim: int = Field(alias="headDim")
    
    class Config:
        populate_by_name = True


class ExpertActivation(BaseModel):
    """Activation data for a single expert."""
    expertId: int = Field(alias="expertId")
    activations: TracedTensor
    
    class Config:
        populate_by_name = True


class MoEState(BaseModel):
    """Detailed MoE layer state."""
    gatingWeights: TracedTensor = Field(alias="gatingWeights")
    selectedExperts: List[List[int]] = Field(alias="selectedExperts")
    expertActivations: List[ExpertActivation] = Field(alias="expertActivations")
    numExperts: int = Field(alias="numExperts")
    topK: int = Field(alias="topK")
    
    class Config:
        populate_by_name = True


class LayerState(BaseModel):
    """
    Complete state of a layer including pre/post activations and layer-specific data.
    This is the new schema as per the ticket requirements.
    """
    layerId: int = Field(alias="layerId")
    layerName: str = Field(alias="layerName")
    layerType: Literal["attention", "moe", "feedforward", "embedding", "normalization", "output"] = Field(alias="layerType")
    inputShape: List[int] = Field(alias="inputShape")
    outputShape: List[int] = Field(alias="outputShape")
    preActivations: Optional[TracedTensor] = Field(None, alias="preActivations")
    postActivations: Optional[TracedTensor] = Field(None, alias="postActivations")
    weights: Optional[TracedTensor] = None
    attentionState: Optional[AttentionState] = Field(None, alias="attentionState")
    moeState: Optional[MoEState] = Field(None, alias="moeState")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        populate_by_name = True


class ForwardStep(BaseModel):
    """
    A single forward pass computation step.
    This is the new schema as per the ticket requirements.
    """
    stepIndex: int = Field(alias="stepIndex")
    layerState: LayerState = Field(alias="layerState")
    description: str
    timingMs: Optional[float] = Field(None, alias="timingMs")
    
    class Config:
        populate_by_name = True


class ModelMetadata(BaseModel):
    """Metadata about the model and execution."""
    vocabSize: int = Field(alias="vocabSize")
    embeddingDim: int = Field(alias="embeddingDim")
    hiddenDim: int = Field(alias="hiddenDim")
    numLayers: int = Field(alias="numLayers")
    tokenizerType: str = Field(alias="tokenizerType")
    totalTimeMs: Optional[float] = Field(None, alias="totalTimeMs")
    
    class Config:
        populate_by_name = True


class ModelRunResponse(BaseModel):
    """
    Complete model run response with forward tracing.
    This is the new schema as per the ticket requirements.
    """
    success: bool
    inputText: str = Field(alias="inputText")
    tokens: List[Token]
    tokenCount: int = Field(alias="tokenCount")
    steps: List[ForwardStep]
    finalLogits: Optional[TracedTensor] = Field(None, alias="finalLogits")
    outputProbabilities: List[OutputProbability] = Field(alias="outputProbabilities")
    metadata: ModelMetadata
    warnings: Optional[List[str]] = None
    
    class Config:
        populate_by_name = True
