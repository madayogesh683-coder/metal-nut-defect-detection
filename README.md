# Metal Nut Defect Detection Using YOLOv8

## 1. Project Overview

Metal Nut Defect Detection is a computer vision project that uses YOLOv8 (You Only Look Once) to automatically identify and localize defects in metal nut images.

The system detects four types of defects:

- Bent
- Color
- Flip
- Scratch

The objective is to support automated visual quality inspection, reduce manual inspection effort, and improve defect identification in manufacturing environments.

## 2. Problem Statement

Manual inspection of metal nuts can be time-consuming and may lead to inconsistent defect identification.

This project proposes an automated defect detection system using deep learning and computer vision to identify visible defects and mark their locations using bounding boxes.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10 | Programming language |
| YOLOv8 | Object detection |
| PyTorch | Deep learning framework |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| MVTec AD | Dataset |
| Ultralytics | Model training and prediction |
| VS Code | Development environment |

## 4. Dataset Description

The project uses the Metal Nut category from the MVTec Anomaly Detection (MVTec AD) dataset.

The dataset contains normal metal nut images and defective samples.

Four defect categories are used:

1. Bent
2. Color
3. Flip
4. Scratch

The dataset was prepared in YOLO format with separate training, validation, and testing directories.

**Prepared dataset distribution:**

- Training images: 284
- Validation images: 23
- Testing images: 28

## 5. Project Architecture

The workflow consists of the following stages:

1. Dataset collection
2. Image preprocessing and annotation conversion
3. Dataset splitting
4. YOLOv8 model training
5. Model evaluation
6. Defect prediction
7. Visualization of detected defects

## 6. Project Structure

```text
metal_nut/
│
├── dataset/
│   ├── metal_nut/
│   └── yolo_metal_nut_4class/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
│
├── src/
│   ├── prepare_data.py
│   ├── verify_data.py
│   ├── check_classes.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── predict_single.py
│
├── runs/
│   └── detect/
│       └── outputs/
│           └── metal_nut_4class/
│               └── weights/
│                   └── best.pt
│
├── outputs/
│   └── metal_nut_predictions/
│
├── reports/
├── models/
├── tests/
├── requirements.txt
└── README.md
```

*This is the logical project structure. Keep the actual folder names and paths consistent with your laptop.*

## 7. Installation and Setup

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd metal_nut
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment (Windows)

```powershell
.venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## 8. Execution Instructions

Run the following commands from the project root.

### Prepare dataset

```bash
python -m src.prepare_data
```

### Verify dataset

```bash
python -m src.verify_data
```

### Check class distribution

```bash
python -m src.check_classes
```

### Train the YOLOv8 model

```bash
python -m src.train
```

### Evaluate the trained model

```bash
python -m src.evaluate
```

### Generate predictions

```bash
python -m src.predict
```

## 9. Model Evaluation Results

The trained YOLOv8 model was evaluated using the prepared test dataset.

| Evaluation Metric | Result |
|---|---:|
| Precision | 86.87% |
| Recall | 64.61% |
| mAP@50 | 79.39% |
| mAP@50–95 | 51.87% |

These results represent the current experimental evaluation of the model.

## 10. Defect Detection Categories

- **Bent:** Identifies bent or deformed metal nuts.
- **Color:** Identifies abnormal color-related defects.
- **Flip:** Identifies incorrectly oriented or flipped nuts.
- **Scratch:** Identifies visible scratches on the nut surface.

The system produces prediction images with detected classes and bounding boxes.

## 11. Limitations

- The prepared test dataset contains only 28 images and 17 defect instances.
- Performance may vary with lighting, image quality, camera angle, and manufacturing conditions.
- Scratch and bent detection require further improvement and validation.
- The model has not yet been validated in a real industrial production environment.

## 12. Future Scope

- Increase the dataset size.
- Improve detection of difficult defect categories.
- Integrate real-time camera-based inspection.
- Deploy the model through a web application.
- Integrate with industrial IoT systems.
- Explore edge deployment using suitable hardware.

## 13. Conclusion

This project demonstrates the application of YOLOv8 for automated metal nut defect detection.

The developed system identifies four defect categories and generates visual predictions with bounding boxes.

The experimental results demonstrate the potential of deep learning-based visual inspection for manufacturing quality control. Further training and real-world validation can improve reliability.

## 14. License and Dataset Reference

Dataset: MVTec Anomaly Detection (MVTec AD)

Official dataset website: https://www.mvtec.com/company/research/datasets/mvtec-ad

Refer to the dataset's license and usage conditions before redistribution.
