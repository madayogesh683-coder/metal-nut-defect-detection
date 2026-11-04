from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "metal_nut_best.pt"

EXPECTED_CLASSES = ["bent", "color", "flip", "scratch"]

def test_trained_model_exists():
    """Check that the trained model file is available."""
    assert MODEL_PATH.exists(), "Trained model file is missing."
    assert MODEL_PATH.stat().st_size > 0

def test_model_loads_and_predicts():
    """Check model loading, class names and inference."""
    model = YOLO(str(MODEL_PATH))

    class_names = list(model.names.values())

    assert class_names == EXPECTED_CLASSES

    # Create a synthetic blank image for an inference smoke test.
    image = np.zeros((416, 416, 3), dtype=np.uint8)

    results = model.predict(
        source=image,
        imgsz=416,
        device="cpu",
        verbose=False
    )

    assert len(results) == 1
    assert results[0].boxes is not None
    