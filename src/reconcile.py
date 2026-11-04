
from pathlib import Path
import csv

GT_DIR = Path("dataset/yolo_metal_nut_4class/labels/test")
PRED_DIR = Path(
    "runs/detect/outputs/metal_nut_reconciliation/labels"
)
REPORT_DIR = Path("reports")

CLASSES = {
    0: "bent",
    1: "color",
    2: "flip",
    3: "scratch"
}

IOU_THRESHOLD = 0.5

def read_labels(path):
    labels = []

    if not path.exists():
        return labels

    for line in path.read_text().splitlines():
        values = line.split()

        if len(values) < 5:
            continue

        class_id = int(values[0])
        box = list(map(float, values[1:5]))
        confidence = float(values[5]) if len(values) > 5 else 1.0

        labels.append((class_id, box, confidence))

    return labels

def calculate_iou(box1, box2):
    x1 = max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
    y1 = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
    x2 = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
    y2 = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def main():
    REPORT_DIR.mkdir(exist_ok=True)

    totals = {
        class_id: {"TP": 0, "FP": 0, "FN": 0}
        for class_id in CLASSES
    }

    image_rows = []

    for gt_file in sorted(GT_DIR.glob("*.txt")):
        pred_file = PRED_DIR / gt_file.name

        actual = read_labels(gt_file)
        predicted = read_labels(pred_file)

        matched_predictions = set()
        matched_actual = set()

        candidates = []

        for gi, (g_class, g_box, _) in enumerate(actual):
            for pi, (p_class, p_box, confidence) in enumerate(predicted):
                if g_class == p_class:
                    iou = calculate_iou(g_box, p_box)
                    if iou >= IOU_THRESHOLD:
                        candidates.append((iou, confidence, gi, pi))

        candidates.sort(reverse=True)

        for iou, confidence, gi, pi in candidates:
            if gi not in matched_actual and pi not in matched_predictions:
                matched_actual.add(gi)
                matched_predictions.add(pi)

        for class_id in CLASSES:
            gt_count = sum(1 for x in actual if x[0] == class_id)
            pred_count = sum(1 for x in predicted if x[0] == class_id)

            tp = sum(
                1 for gi in matched_actual
                if actual[gi][0] == class_id
            )

            fp = pred_count - tp
            fn = gt_count - tp

            totals[class_id]["TP"] += tp
            totals[class_id]["FP"] += fp
            totals[class_id]["FN"] += fn

            image_rows.append({
                "image": gt_file.stem,
                "class": CLASSES[class_id],
                "actual": gt_count,
                "predicted": pred_count,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn
            })

    with open(
        REPORT_DIR / "reconciliation_per_image.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=image_rows[0].keys())
        writer.writeheader()
        writer.writerows(image_rows)

    summary_rows = []

    for class_id, name in CLASSES.items():
        values = totals[class_id]
        tp, fp, fn = values["TP"], values["FP"], values["FN"]

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        summary_rows.append({
            "class": name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        })

    with open(
        REPORT_DIR / "reconciliation_summary.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Reconciliation completed!")
    print("Reports saved in:", REPORT_DIR.resolve())
    print("\nClass-wise results:")

    for row in summary_rows:
        print(row)

if __name__ == "__main__":
    main()
    