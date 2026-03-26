import os
from pathlib import Path
import uui_iris_predictor
from uui_iris_predictor.version import __version__

PACKAGE_ROOT = Path(uui_iris_predictor.__file__).resolve().parent

PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", __version__)
TRAINED_MODEL_DIR = Path(os.environ.get("TRAINED_MODEL_DIR", PACKAGE_ROOT / "trained_models"))
