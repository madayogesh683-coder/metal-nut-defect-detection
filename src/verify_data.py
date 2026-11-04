
from pathlib import Path

from src.evaluate import ROOT
ROOT = Path("dataset/yolo_metal_nut_4class")

for split in ["train", "val", "test"]:
    images_dir = ROOT / "images" / split
    labels_dir = ROOT / "labels" / split

    images = list(images_dir.glob("*"))
    labels = list(labels_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing = image_stems - label_stems
    extra = label_stems - image_stems

    nonempty = sum(
        1 for p in labels
        if p.read_text(encoding="utf-8").strip()
    )

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Non-empty labels:", nonempty)
    print("Missing labels:", len(missing))
    print("Extra labels:", len(extra))

    if missing or extra or len(images) != len(labels):
        print("STATUS: FAILED")
    else:
        print("STATUS: PASSED")

print("\nVerification completed.")