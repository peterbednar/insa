from pathlib import Path
import uui_iris_predictor

PACKAGE_ROOT = Path(uui_iris_predictor.__file__).resolve().parent

PIPELINE_VERSION="0.1.0"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
