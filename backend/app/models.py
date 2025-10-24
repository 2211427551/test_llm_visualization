from pydantic import BaseModel, Field
from typing import List, Optional


class Token(BaseModel):
    text: str
    id: int


class LayerData(BaseModel):
    layerId: int = Field(alias="layerId")
    layerName: str = Field(alias="layerName")
    inputShape: List[int] = Field(alias="inputShape")
    outputShape: List[int] = Field(alias="outputShape")
    activations: Optional[List[List[float]]] = None
    weights: Optional[List[List[float]]] = None
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
