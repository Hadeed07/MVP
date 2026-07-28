from pathlib import Path

THIS_FILE = Path(__file__)

BASE_DIR = THIS_FILE.parent

MODELS_DIR = BASE_DIR / 'models'
YOLO_WEIGHTS = MODELS_DIR / 'best_obb.pt'