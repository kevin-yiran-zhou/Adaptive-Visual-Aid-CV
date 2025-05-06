import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import time

# ----- Device -----
device = torch.device("cpu")
print("Using device:", device)

# ----- Load Hugging Face SegFormer Model -----
# B5: mIoU 49.3 => too slow
# model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
# B4: 
# model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
# B3: 
# model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
# B2: mIoU 44.0
model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
# B1: mIoU 39.7 => accuracy not good
# model_name = "nvidia/segformer-b1-finetuned-ade-512-512"
# B0: mIoU 37.4 => accuracy too bad
# model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
model.eval()

# ----- ADE20K label map and color map -----
id2label = model.config.id2label

# ADE20K color palette (150 entries) – shortened version for demo
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

def draw_legend(segmap):
    unique_classes = np.unique(segmap)
    legend_patches = []
    for cls_id in unique_classes:
        if cls_id < len(id2label):
            name = id2label[cls_id]
            color = np.array(ADE20K_COLORS[cls_id]) / 255
            patch = plt.Rectangle((0, 0), 1, 1, color=color, label=name)
            legend_patches.append(patch)
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

# ----- Inference on all JPEG images in ./images -----
image_files = sorted(glob.glob("/home/kevin-zhou/Desktop/UMich/WeilandLab/Adaptive-Visual-Aid-CV/images/*.jpeg"))

for img_path in image_files:
    print(f"Processing {img_path}...")
    start_time = time.time()
    image = Image.open(img_path).convert("RGB")
    ### resize image ###
    # resize_factor = 5
    # new_size = (int(image.size[0] // resize_factor), int(image.size[1] // resize_factor))
    # image = image.resize(new_size, Image.BILINEAR)
    ####################

    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s | FPS: {1/inference_time:.2f}")
    overlay = overlay_segmentation(image, pred)

    plt.figure(figsize=(10, 5))
    plt.imshow(overlay)
    plt.title("SegFormer (ADE20K) Overlay")
    plt.axis("off")
    draw_legend(pred)
    plt.tight_layout()
    plt.show()