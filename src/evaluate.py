
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = (
    "runs/detect/outputs/metal_nut_4class/weights/best.pt"
)

DATA_PATH = "dataset/yolo_metal_nut_4class/data.yaml"

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading trained four-class model...")

    model = YOLO(MODEL_PATH)

    print("\nEvaluating on the TEST dataset...")

    metrics = model.val(
        data=DATA_PATH,
        split="test",
        imgsz=416,
        batch=4,
        device="cpu",
        plots=True,
        project="outputs",
        name="metal_nut_4class_test_eval",
        exist_ok=True
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
    