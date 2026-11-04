
from pathlib import Path
import cv2
import numpy as np

OUTPUT = Path("sample_data")
IMAGE_DIR = OUTPUT / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def create_nut_image(filename, defect=False):
    image = np.full((416, 416, 3), 65, dtype=np.uint8)

    center = (208, 208)

    # Draw a metal washer-like object
    cv2.circle(image, center, 115, (160, 160, 160), -1)
    cv2.circle(image, center, 48, (65, 65, 65), -1)

    # Add highlights for a metallic appearance
    cv2.circle(image, (175, 165), 18, (205, 205, 205), -1)

    if defect:
        # Add a visible scratch
        cv2.line(
            image,
            (125, 180),
            (190, 225),
            (20, 20, 20),
            7
        )

    path = IMAGE_DIR / filename
    cv2.imwrite(str(path), image)

def main():
    create_nut_image("sample_normal.png")
    create_nut_image("sample_scratch.png", defect=True)

    print("Synthetic sample images generated successfully.")
    print(f"Output directory: {IMAGE_DIR.resolve()}")

if __name__ == "__main__":
    main()
    