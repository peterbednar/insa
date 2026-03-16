from pathlib import Path
import iris_predictor

PACKAGE_ROOT = Path(iris_predictor.__file__).resolve().parent

PIPELINE_VERSION="0.1.0"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DEFAULT_PIPELINE_NAME = f"pipeline_{PIPELINE_VERSION}.pkl"
