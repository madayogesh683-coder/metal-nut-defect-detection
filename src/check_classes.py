from pathlib import Path
from collections import Counter

ROOT = Path("dataset/yolo_metal_nut_4class/labels")

class_names = {
    0: "bent",
    1: "color",
    2: "flip",
    3: "scratch"
}

for split in ["train", "val", "test"]:
    counter = Counter()
    invalid = 0

    folder = ROOT / split

    for file in folder.glob("*.txt"):
        for line in file.read_text().splitlines():
            if not line.strip():
                continue

            parts = line.split()

            try:
                class_id = int(parts[0])
                if class_id not in class_names:
                    invalid += 1
                else:
                    counter[class_id] += 1
            except (ValueError, IndexError):
                invalid += 1

    print(f"\n{split.upper()} CLASS DISTRIBUTION")

    for class_id, name in class_names.items():
        print(f"{name}: {counter[class_id]}")

    print("Invalid labels:", invalid)
    