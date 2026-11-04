from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "yolo_metal_nut"

for split in ["train", "val", "test"]:
    folder = DATASET / "images" / split

    counts = Counter(
        image.name.split("_")[0]
        for image in folder.glob("*")
    )

    print(f"\n{split.upper()} DISTRIBUTION")
    for category, count in sorted(counts.items()):
        print(f"{category}: {count}")
        