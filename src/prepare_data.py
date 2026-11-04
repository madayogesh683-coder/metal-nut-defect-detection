from pathlib import Path
import random
import shutil
import cv2
import yaml

SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

ROOT = Path("dataset/metal_nut")
OUTPUT = Path("dataset/yolo_metal_nut_4class")

DEFECT_TYPES = {
    "bent": 0,
    "color": 1,
    "flip": 2,
    "scratch": 3
}

def get_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

def split_items(items):
    items = list(items)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:]
    }

def mask_to_yolo(mask_path, width, height, class_id):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    if mask.shape != (height, width):
        raise ValueError(f"Image/mask size mismatch: {mask_path}")

    binary = (mask > 0).astype("uint8")

    if cv2.countNonZero(binary) == 0:
        return []

    x, y, box_width, box_height = cv2.boundingRect(binary)

    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height

    normalized_width = box_width / width
    normalized_height = box_height / height

    annotation = (
        f"{class_id} "
        f"{center_x:.6f} "
        f"{center_y:.6f} "
        f"{normalized_width:.6f} "
        f"{normalized_height:.6f}"
    )

    return [annotation]

def create_output_folders():
    for split in ["train", "val", "test"]:
        (OUTPUT / "images" / split).mkdir(
            parents=True, exist_ok=True
        )
        (OUTPUT / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )

def save_sample(image_path, split, annotations, category):
    unique_name = f"{category}_{image_path.name}"

    image_out = OUTPUT / "images" / split / unique_name

    label_out = (
        OUTPUT / "labels" / split /
        f"{category}_{image_path.stem}.txt"
    )

    shutil.copyfile(image_path, image_out)

    label_out.write_text(
        "\n".join(annotations),
        encoding="utf-8"
    )

def main():
    random.seed(SEED)

    if not ROOT.is_dir():
        raise FileNotFoundError(f"Dataset not found: {ROOT}")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    create_output_folders()

    counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    print("\nPreparing four-class defective images...")

    for defect, class_id in DEFECT_TYPES.items():

        image_folder = ROOT / "test" / defect
        mask_folder = ROOT / "ground_truth" / defect

        images = get_images(image_folder)
        groups = split_items(images)

        print(f"\n{defect.upper()} - Class ID: {class_id}")
        print("Total images:", len(images))

        for split, samples in groups.items():

            for image_path in samples:

                mask_path = (
                    mask_folder /
                    f"{image_path.stem}_mask{image_path.suffix}"
                )

                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Missing mask: {mask_path}"
                    )

                image = cv2.imread(str(image_path))

                if image is None:
                    raise ValueError(
                        f"Cannot read image: {image_path}"
                    )

                height, width = image.shape[:2]

                annotations = mask_to_yolo(
                    mask_path,
                    width,
                    height,
                    class_id
                )

                if not annotations:
                    raise ValueError(
                        f"No defect region found: {mask_path}"
                    )

                save_sample(
                    image_path,
                    split,
                    annotations,
                    defect
                )

                counts[split] += 1

    print("\nPreparing normal images...")

    train_good = get_images(ROOT / "train" / "good")

    for image_path in train_good:
        save_sample(image_path, "train", [], "good")
        counts["train"] += 1

    test_good = get_images(ROOT / "test" / "good")
    random.shuffle(test_good)

    midpoint = len(test_good) // 2

    normal_groups = {
        "val": test_good[:midpoint],
        "test": test_good[midpoint:]
    }

    for split, samples in normal_groups.items():

        for image_path in samples:
            save_sample(image_path, split, [], "good_test")
            counts[split] += 1

    data_config = {
        "path": str(OUTPUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "bent",
            1: "color",
            2: "flip",
            3: "scratch"
        }
    }

    with open(OUTPUT / "data.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(data_config, file, sort_keys=False)

    print("\n========== DATA PREPARATION COMPLETE ==========")

    for split, count in counts.items():
        print(f"{split}: {count} images")

    print("\nYOLO dataset:", OUTPUT.resolve())
    print("Configuration:", OUTPUT / "data.yaml")

if __name__ == "__main__":
    main()
    