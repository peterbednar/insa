import pandas as pd
from fastapi import FastAPI, Request
from collections import defaultdict
from uui_iris_predictor.data_manager import load_pipeline
from uui_iris_predictor.version import __version__
from uui_iris_predictor.config import PIPELINE_VERSION
from uui_iris_predictor.data_model import RpcRequest, RpcResponse, RpcError, Result, Prediction, Error

app = FastAPI()

@app.get("/")
async def info():
    return {"version": __version__,
            "pipeline_version": PIPELINE_VERSION}

pipelines = defaultdict(load_pipeline)

def predict(params):
    pipe = pipelines[params.pipeline]
    X = pd.DataFrame.from_records([record.model_dump() for record in params.data])

    labels = pipe.predict(X)
    probs = pipe.predict_proba(X)
    probs = [dict(zip(pipe.steps[-1][1].classes_, p)) for p in probs]

    predictions = [Prediction(label=label, proba=proba) for label, proba in zip(labels, probs)]
    return Result(
        pipeline=params.pipeline,
        predictions=predictions
    )

@app.post("/api/v2/rpc")
async def json_rpc(request: Request) -> RpcResponse | RpcError:
    rpc_request = None
    try:
        rpc_request = RpcRequest(** await request.json())
        result = predict(rpc_request.params)
        return RpcResponse(
            jsonrpc="2.0",
            result=result,
            id=rpc_request.id
        )
    except Exception as e:
        return RpcError(
            jsonrpc="2.0",
            error=Error(code=-32000, message=str(e)),
            id=rpc_request.id if rpc_request is not None else None
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True # auto-reload for development
    )
