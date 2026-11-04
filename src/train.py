
from ultralytics import YOLO
from pathlib import Path

DATA_CONFIG = "dataset/yolo_metal_nut_4class/data.yaml"

def main():
    if not Path(DATA_CONFIG).exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_CONFIG}"
        )

    print("Loading pretrained YOLOv8n model...")

    model = YOLO("yolov8n.pt")

    print("Starting four-class metal nut defect training...")

    results = model.train(
        data=DATA_CONFIG,
        epochs=50,
        imgsz=416,
        batch=4,
        patience=15,
        device="cpu",
        workers=0,
        project="outputs",
        name="metal_nut_4class",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    print("\nTraining completed!")
    print("Best model:")
    print("outputs/metal_nut_4class/weights/best.pt")

if __name__ == "__main__":
    main()
    