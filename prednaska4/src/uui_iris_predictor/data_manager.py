import joblib
from uui_iris_predictor.config import TRAINED_MODEL_DIR, DEFAULT_PIPELINE_FILENAME

def save_pipeline(pipeline, filename=None):
    if filename is None:
        filename = DEFAULT_PIPELINE_FILENAME
    joblib.dump(pipeline, TRAINED_MODEL_DIR / filename)

def load_pipeline(filename=None):
    if filename is None:
        filename = DEFAULT_PIPELINE_FILENAME
    return joblib.load(TRAINED_MODEL_DIR / filename)

def pipeline_exists(filename=None):
    if filename is None:
        filename = DEFAULT_PIPELINE_FILENAME
    return (TRAINED_MODEL_DIR / filename).is_file()
