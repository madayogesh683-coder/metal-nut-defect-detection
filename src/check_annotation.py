from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset/yolo_metal_nut/images/test"
LABEL_DIR = ROOT / "dataset/yolo_metal_nut/labels/test"

# Find colored image, with or without category prefix
matches = list(IMAGE_DIR.glob("*color*.*"))

if not matches:
    raise FileNotFoundError("color image not found")

image_path = matches[0]
label_path = LABEL_DIR / f"{image_path.stem}.txt"

if not label_path.exists():
    raise FileNotFoundError(f"Label not found: {label_path}")

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Could not read image")

height, width = image.shape[:2]

with open(label_path, "r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        values = list(map(float, line.split()))

        class_id, xc, yc, bw, bh = values

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            "Ground Truth Defect",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

output_path = ROOT / "outputs" / "color_annotation_check.jpg"

cv2.imwrite(str(output_path), image)

print("Annotation visualization saved:")
print(output_path)
