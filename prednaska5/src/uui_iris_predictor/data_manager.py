import joblib
from uui_iris_predictor.config import TRAINED_MODEL_DIR, PIPELINE_VERSION

def pipeline_path(version):
    return TRAINED_MODEL_DIR / f"pipeline_{version}.pkl"

def save_pipeline(pipeline, version=PIPELINE_VERSION):
    joblib.dump(pipeline, pipeline_path(version))

def load_pipeline(version=PIPELINE_VERSION):
    return joblib.load(pipeline_path(version))

def pipeline_exists(version=PIPELINE_VERSION):
    return pipeline_path(version).is_file()
