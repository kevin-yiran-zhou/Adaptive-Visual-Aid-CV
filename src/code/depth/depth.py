import os
import glob
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForDepthEstimation
import torch
import numpy as np

# Load model and processor
# Model	                  HuggingFace ID	            Notes
# DPT-Large	              Intel/dpt-large	            Best accuracy, heaviest (~439M params)
# DPT-Hybrid	          Intel/dpt-hybrid-midas	    Great balance
# DPT-SwinV2-Tiny-256	  Intel/dpt-swinv2-tiny-256	    Lightweight, optimized for speed
# DPT-SwinV2-Tiny-384	  Intel/dpt-swinv2-tiny-384	    Better quality than 256, still fast
# LeReS	                  nielsr/dpt-depth-estimation	Good accuracy, slower
# ZoeDepth	              isl-org/ZoeDepth	            Newer, high-accuracy, works well in indoor/outdoor


model_name = "Intel/dpt-large"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name)
model.eval()

# Path to your images
image_files = sorted(glob.glob("/home/kevin-zhou/Desktop/UMich/WeilandLab/Adaptive-Visual-Aid-CV/images/*.png"))

for img_path in image_files:
    print(f"Processing {img_path}...")
    start_time = time.time()
    image = Image.open(img_path).convert("RGB")
    orig_width, orig_height = image.size

    # Process image
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth[0].squeeze().cpu().numpy()

    # Normalize and resize depth map
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
    depth_vis_resized = cv2.resize(depth_vis, (orig_width, orig_height))

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s | FPS: {1/inference_time:.2f}")
    # Plot side-by-side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(depth_vis_resized, cmap="plasma")
    plt.title("Estimated Depth (Resized)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
