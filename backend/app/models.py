from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Token(BaseModel):
    text: str
    id: int


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
