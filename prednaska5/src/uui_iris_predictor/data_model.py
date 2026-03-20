from uui_iris_predictor.config import PIPELINE_VERSION
from typing import Literal
from pydantic import BaseModel

class Record(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Params(BaseModel):
    data: list[Record]
    pipeline: str = PIPELINE_VERSION

class Result(BaseModel):
    y_pred: list[int] | None
    y_proba: list[list[float]] | None
    y_log_proba: list[list[float]] | None

class Error(BaseModel):
    code: int
    message: str
    data: dict

class RpcRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    method: Literal["pipeline/predict"]
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
