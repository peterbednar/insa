from pydantic import BaseModel
from typing import Literal, Dict, List
from uui_iris_predictor.config import PIPELINE_VERSION

class Record(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Params(BaseModel):
    data: List[Record]
    pipeline: str = PIPELINE_VERSION

class Prediction(BaseModel):
    label: str | int
    proba: Dict[str | int, float] | None = None

class Result(BaseModel):
    predictions: List[Prediction]
    pipeline: str

class Error(BaseModel):
    code: int
    message: str
    data: dict

class RpcRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    method: Literal["pipeline/predict", "pipeline/predict_proba"]
    params: Params
    id: int | None = None

class RpcResponse(BaseModel):
    jsonrpc: Literal["2.0"]
    result: Result
    id: int | None = None

class RpcError(BaseModel):
    jsonrpc: Literal["2.0"]
    error: Error
    id: int | None = None
