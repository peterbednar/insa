import os
import logging
from pathlib import Path
import uui_iris_predictor
from uui_iris_predictor.version import __version__

PACKAGE_ROOT = Path(uui_iris_predictor.__file__).resolve().parent

PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", __version__)
TRAINED_MODEL_DIR = Path(os.getenv("TRAINED_MODEL_DIR", PACKAGE_ROOT / "trained_models"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)

logging.basicConfig(level=LOG_LEVEL)

def get_logger():
    return logging.getLogger("uui_iris_predictor")
