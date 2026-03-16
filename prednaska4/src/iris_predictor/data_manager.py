import joblib
from iris_predictor.config import TRAINED_MODEL_DIR, DEFAULT_PIPELINE_NAME

def save_pipeline(pipeline, filename=None):
    if filename is None:
        filename = DEFAULT_PIPELINE_NAME
    joblib.dump(pipeline, TRAINED_MODEL_DIR / filename)

def load_pipeline(file_name=None):
    if filename is None:
        filename = DEFAULT_PIPELINE_NAME
    return joblib.load(TRAINED_MODEL_DIR / filename)
