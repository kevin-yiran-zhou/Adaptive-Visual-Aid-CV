import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline

# Load depth estimation pipeline (small model for speed)
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# Path to your images
image_dir = "/home/kevin-zhou/Desktop/UMich/WeilandLab/Adaptive-Visual-Aid-CV/images"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpeg")]

for file_name in image_files:
    img_path = os.path.join(image_dir, file_name)
    image = Image.open(img_path).convert("RGB")
    orig_width, orig_height = image.size

    # Predict depth
    result = pipe(image)
    depth = result["depth"]
    depth_np = np.array(depth)

    # Normalize and resize depth map
    depth_vis = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    depth_resized = np.array(depth_vis.resize((orig_width, orig_height), Image.BILINEAR))

    # Plot original and depth map
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Original - {file_name}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(depth_resized, cmap="plasma")
    plt.title("Depth Anything V2 (HF)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()