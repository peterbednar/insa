from prometheus_client import Counter, Histogram

rpc_requests_total = Counter(
    "rpc_requests_total",
    "Total number of RPC requests",
    ["method", "status"]
)

rpc_request_duration_seconds = Histogram(
    "rpc_request_duration_seconds",
    "RPC request latency",
    ["method"]
)

predictions_total = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["pipeline"]
)

prediction_distribution = Histogram(
    "prediction_distribution",
    "Distribution of predicted labels",
    ["pipeline", "label"],
    buckets=[p / 100 for p in range(0, 100, 5)]
)
