import os
import glob
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor, AutoModelForDepthEstimation,
    SegformerFeatureExtractor, SegformerForSemanticSegmentation,
    pipeline
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device:", device)

# DPT model: https://arxiv.org/abs/1907.01341v3
# "Intel/dpt-large"   "Intel/dpt-hybrid-midas"   "Intel/dpt-swinv2-tiny-256"
dpt_model_name = "Intel/dpt-hybrid-midas"
dpt_processor = AutoProcessor.from_pretrained(dpt_model_name)
dpt_model = AutoModelForDepthEstimation.from_pretrained(dpt_model_name).to(device)
dpt_model.eval()

# Depth Anything pipeline: https://depth-anything.github.io/ https://depth-anything-v2.github.io/ https://promptda.github.io/ 
# "depth-anything/Depth-Anything-V2-Small-hf"   "LiheYoung/depth-anything-small-hf"     (Small/Base/Large)
depth_anything_model_name = "depth-anything/Depth-Anything-V2-Base-hf"
depth_anything_pipe = pipeline("depth-estimation", model=depth_anything_model_name, device=0 if torch.cuda.is_available() else -1)

# UniDepth: https://github.com/lpiccinelli-eth/UniDepth

# SegFormer: https://arxiv.org/abs/2105.15203 
# "nvidia/segformer-b5-finetuned-ade-640-640"   "nvidia/segformer-b2-finetuned-ade-512-512"     "nvidia/segformer-b1-finetuned-ade-512-512"     "nvidia/segformer-b0-finetuned-ade-512-512"
seg_model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
seg_extractor = SegformerFeatureExtractor.from_pretrained(seg_model_name)
seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model_name).to(device)
seg_model.eval()
id2label = seg_model.config.id2label

ADE20K_COLORS = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
    [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
    [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
    [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
    [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
    [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]
])

def decode_segmap(segmap):
    h, w = segmap.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(ADE20K_COLORS)):
        seg_rgb[segmap == i] = ADE20K_COLORS[i]
    return seg_rgb

def overlay_segmentation(image_pil, segmap, alpha=0.6):
    seg_rgb = decode_segmap(segmap)
    seg_img = Image.fromarray(seg_rgb).resize(image_pil.size)
    return Image.blend(image_pil.convert("RGB"), seg_img.convert("RGB"), alpha)

def draw_segmentation_legend(ax, segmap):
    unique_classes = np.unique(segmap)
    legend_patches = []
    for cls_id in unique_classes:
        if cls_id < len(id2label):
            name = id2label[cls_id]
            color = np.array(ADE20K_COLORS[cls_id]) / 255
            patch = plt.Rectangle((0, 0), 1, 1, color=color, label=name)
            legend_patches.append(patch)
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

png_files = sorted(glob.glob("/home/kevin-zhou/Desktop/UMich/WeilandLab/Adaptive-Visual-Aid-CV/images/*.png"))
jpeg_files = sorted(glob.glob("/home/kevin-zhou/Desktop/UMich/WeilandLab/Adaptive-Visual-Aid-CV/images/*.jpeg"))
image_files = png_files + jpeg_files

for img_path in png_files:
    print(f"\nProcessing {img_path}...")
    image = ImageOps.exif_transpose(Image.open(img_path).convert("RGB"))
    orig_width, orig_height = image.size
    print(f"Original resolution: {orig_width} x {orig_height}")
    scale = 640 / max(orig_width, orig_height)
    new_size = (int(orig_width * scale), int(orig_height * scale))
    image = image.resize(new_size, Image.BILINEAR)
    print(f"Resized resolution: {image.size[0]} x {image.size[1]}")

    # === Segmentation ===
    t1 = time.time()
    seg_inputs = seg_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        seg_outputs = seg_model(**seg_inputs)
        seg_pred = torch.argmax(seg_outputs.logits, dim=1).squeeze().cpu().numpy()
    seg_time = time.time() - t1
    print(f"Segmentation time: {seg_time:.2f}s")

    # === DPT Depth ===
    t2 = time.time()
    dpt_inputs = dpt_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        dpt_outputs = dpt_model(**dpt_inputs)
        dpt_depth = dpt_outputs.predicted_depth[0].squeeze().cpu().numpy()
    dpt_time = time.time() - t2
    print(f"DPT depth time: {dpt_time:.2f}s")
    dpt_depth_vis = (dpt_depth - dpt_depth.min()) / (dpt_depth.max() - dpt_depth.min())
    dpt_depth_resized = cv2.resize(dpt_depth_vis, (orig_width, orig_height))

    # === Depth Anything ===
    t3 = time.time()
    depth_anything_result = depth_anything_pipe(image)
    da_time = time.time() - t3
    print(f"Depth Anything time: {da_time:.2f}s")
    depth_anything = np.array(depth_anything_result["depth"])
    depth_anything_vis = (depth_anything - depth_anything.min()) / (depth_anything.max() - depth_anything.min())
    depth_anything_resized = cv2.resize(depth_anything_vis, (orig_width, orig_height))

    # === Plotting ===
    overlay_img = overlay_segmentation(image, seg_pred)
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(overlay_img)
    axs[1].set_title("Segmentation")
    axs[1].axis("off")
    draw_segmentation_legend(axs[1], seg_pred)

    im1 = axs[2].imshow(dpt_depth_resized, cmap="plasma")
    axs[2].set_title("DPT Depth")
    axs[2].axis("off")
    plt.colorbar(im1, ax=axs[2], fraction=0.046, pad=0.04)

    im2 = axs[3].imshow(depth_anything_resized, cmap="plasma")
    axs[3].set_title("Depth Anything")
    axs[3].axis("off")
    plt.colorbar(im2, ax=axs[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
