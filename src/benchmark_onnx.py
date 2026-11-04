
import time
import glob
import numpy as np
import onnxruntime as ort
import cv2

MODELS = {
    "416x416": ("models/best_416.onnx", 416),
    "320x320": ("models/best_320.onnx", 320),
}

images = glob.glob("sample_data/images/*")

if not images:
    raise FileNotFoundError("No sample images found in sample_data/images")

for label, (model_path, image_size) in MODELS.items():
    print(f"\nBenchmarking {label} model...")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    dummy = np.zeros(
        (1, 3, image_size, image_size),
        dtype=np.float32
    )

    for _ in range(5):
        session.run(None, {input_name: dummy})

    times = []

    for image_path in images:
        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(image, (image_size, image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

        start = time.perf_counter()
        session.run(None, {input_name: image})
        end = time.perf_counter()

        times.append((end - start) * 1000)

    if not times:
        raise RuntimeError(f"No valid images for {label}")

    average = np.mean(times)

    print(f"\n{label} RESULTS")
    print(f"Images processed: {len(times)}")
    print(f"Average latency: {average:.2f} ms")
    print(f"Minimum latency: {np.min(times):.2f} ms")
    print(f"Maximum latency: {np.max(times):.2f} ms")
    print(f"FPS: {1000 / average:.2f}")
