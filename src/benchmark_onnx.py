import time
import glob
import numpy as np
import onnxruntime as ort
import cv2

MODEL_PATH = "models/best.onnx"
IMAGE_SIZE = 416

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

images = glob.glob("sample_data/images/*")

if not images:
    raise FileNotFoundError("No sample images found!")

times = []

# Warm-up
dummy = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

for _ in range(5):
    session.run(None, {input_name: dummy})

# Benchmark
for image_path in images:
    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

    start = time.perf_counter()
    session.run(None, {input_name: image})
    end = time.perf_counter()

    times.append((end - start) * 1000)

if not times:
    raise RuntimeError("No valid images found.")

print("\nONNX CPU BENCHMARK RESULTS")
print("--------------------------")
print(f"Images processed: {len(times)}")
print(f"Average latency: {np.mean(times):.2f} ms")
print(f"Minimum latency: {np.min(times):.2f} ms")
print(f"Maximum latency: {np.max(times):.2f} ms")
print(f"FPS: {1000 / np.mean(times):.2f}")
