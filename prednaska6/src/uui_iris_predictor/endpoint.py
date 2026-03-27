import time
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from prometheus_client import make_asgi_app
from collections import defaultdict
from uui_iris_predictor.data_manager import load_pipeline
from uui_iris_predictor.version import __version__
from uui_iris_predictor.config import PIPELINE_VERSION, get_logger
from uui_iris_predictor.data_model import RpcRequest, RpcResponse, RpcError, Result, Prediction, Error
from uui_iris_predictor.metrics import rpc_requests_total, rpc_request_duration_seconds, predictions_total, prediction_distribution

log = get_logger()

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

    model = pipe[-1]
    probs = pipe.predict_proba(X)
    probs = [dict(zip(model.classes_, p)) for p in probs]

    predictions = [Prediction(label=label, proba=proba) for label, proba in zip(labels, probs)]
    return Result(
        pipeline=params.pipeline,
        predictions=predictions
    )

async def json_rpc(request: RpcRequest) -> RpcResponse | RpcError:
    try:
        log.debug(f"JSON RPC call {request.method}:{request.params.pipeline} with {len(request.params.data)} records")
        result = await run_in_threadpool(predict, request.params)
        return RpcResponse(
            jsonrpc="2.0",
            result=result,
            id=request.id
        )
    except Exception as e:
        log.exception(e)
        return RpcError(
            jsonrpc="2.0",
            error=Error(code=-32000, message=str(e)),
            id=request.id
        )

@app.post("/api/v2/rpc")
async def json_rpc_with_metrics(request: RpcRequest) -> RpcResponse | RpcRequest:
    time_start = time.time()
    response = await json_rpc(request)
    duration = time.time() - time_start

    rpc_request_duration_seconds.labels(
        method=request.method
    ).observe(duration)

    if isinstance(response, RpcResponse):
        result = response.result

        rpc_requests_total.labels(
            method=request.method,
            status="success"
        ).inc()

        predictions_total.labels(
            pipeline=result.pipeline
        ).inc(len(result.predictions))

        for p in result.predictions:
            if p.proba is not None:
                for label, proba in p.proba.items():
                    prediction_distribution.labels(
                        pipeline=result.pipeline,
                        label=label
                    ).observe(proba)
    else:
        rpc_requests_total.labels(
            method=request.method,
            status="error"
        ).inc()

    return response

app.mount("/metrics", make_asgi_app())

if __name__ == "__main__":
    log.info(f"Starting Iris predictor {__version__}, default pipeline: {PIPELINE_VERSION}")
    pipelines[PIPELINE_VERSION] # precache detault pipeline

    import uvicorn
    uvicorn.run(
        "src.uui_iris_predictor.endpoint:app",
        host="0.0.0.0",
        port=8000
    )
