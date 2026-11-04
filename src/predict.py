from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
ROOT / "runs" / "detect" / "outputs" / "metal_nut_4class"
/ "weights" / "best.pt"
)

TEST_IMAGES = (
ROOT / "dataset" / "yolo_metal_nut_4class"
/ "images" / "test"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

print("Running predictions on test images...")

results = model.predict(
source=str(TEST_IMAGES),
imgsz=416,
conf=0.25,
device="cpu",
save=True,
project=str(ROOT / "runs" / "detect" / "outputs" / "metal_nut_4class"),
name="metal_nut_predictions_v2",
exist_ok=True
)

print("\nPrediction completed!")
print("Total images processed:", len(results))
print("Predictions saved in:")
print(ROOT / "runs" / "detect" / "outputs" / "metal_nut_4class" / "metal_nut_predictions_v2")
