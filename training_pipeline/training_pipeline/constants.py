from enum import Enum
from pathlib import Path


class Scope(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    INFERENCE = "inference"

CACHE_DIR = Path.home() / ".cache" / "ray"