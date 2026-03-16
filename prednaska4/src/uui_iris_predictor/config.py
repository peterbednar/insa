from pathlib import Path
import uui_iris_predictor

PACKAGE_ROOT = Path(uui_iris_predictor.__file__).resolve().parent

PIPELINE_VERSION="0.1.0"
DEFAULT_PIPELINE_FILENAME = f"pipeline_{PIPELINE_VERSION}.pkl"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
