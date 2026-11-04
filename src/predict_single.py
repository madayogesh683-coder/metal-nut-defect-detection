from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = Path(r"C:\Users\maday\Downloads\scratch sample.jpg")

# Find the trained model
possible_models = [
    ROOT / "outputs" / "metal_nut_yolov8_v2" / "weights" / "best.pt",
    ROOT / "outputs" / "metal_nut_yolov8" / "weights" / "best.pt",
]

MODEL_PATH = next(
    (p for p in possible_models if p.exists()), None
)

if MODEL_PATH is None:
    raise FileNotFoundError("Trained best.pt model not found.")

if not IMAGE_PATH.exists():
    raise FileNotFoundError("scratch sample.jpg not found.")

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(IMAGE_PATH),
    imgsz=416,
    conf=0.10,
    device="cpu",
    save=True,
    project=str(ROOT / "outputs"),
    name="single_scratch_prediction",
    exist_ok=True
)

print("Prediction completed!")
print("Results saved in:")
print(ROOT / "outputs" / "single_scratch_prediction")
