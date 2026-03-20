from fastapi import FastAPI
from uui_iris_predictor.version import __version__
from uui_iris_predictor.config import PIPELINE_VERSION
from uui_iris_predictor.data_model import RpcRequest, RpcResponse, RpcError

app = FastAPI()

@app.get("/")
def info():
    return {"version": __version__,
            "pipeline_version": PIPELINE_VERSION}

@app.post("/api/v2/rpc")
def json_rpc(request: RpcRequest) -> RpcResponse | RpcError:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True # auto-reload for development
    )
