import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
from transformers import AutoFeatureExtractor, AutoModelForSemanticSegmentation

# ----- Device -----
device = torch.device("cpu")
print("Using device:", device)

# ----- Load DeepLabV3 model trained on Cityscapes -----
feature_extractor = AutoFeatureExtractor.from_pretrained("valhalla/deeplabv3-cityscapes")
model = AutoModelForSemanticSegmentation.from_pretrained("valhalla/deeplabv3-cityscapes").to(device)
model.eval()

# ----- Class labels and color map (Cityscapes, 19 classes) -----
id2label = model.config.id2label

CITYSCAPES_COLORS = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32]     # bicycle
])

def decode_segmap(segmap):
    h, w = segmap.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(CITYSCAPES_COLORS)):
        seg_rgb[segmap == i] = CITYSCAPES_COLORS[i]
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
            color = CITYSCAPES_COLORS[cls_id] / 255
            patch = plt.Rectangle((0, 0), 1, 1, color=color, label=name)
            legend_patches.append(patch)
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

# ----- Inference on all JPEG images in ./images -----
image_files = sorted(glob.glob("images/*.jpeg"))

for img_path in image_files:
    print(f"Processing {img_path}...")
    start_time = time.time()
    image = Image.open(img_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s | FPS: {1/inference_time:.2f}")
    overlay = overlay_segmentation(image, pred)

    plt.figure(figsize=(10, 5))
    plt.imshow(overlay)
    plt.title("DeepLabV3 (Cityscapes) Overlay")
    plt.axis("off")
    draw_legend(pred)
    plt.tight_layout()
    plt.show()
