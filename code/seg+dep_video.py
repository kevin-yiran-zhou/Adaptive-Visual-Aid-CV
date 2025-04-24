import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor, AutoModelForDepthEstimation,
    SegformerFeatureExtractor, SegformerForSemanticSegmentation,
    pipeline
)

# === Config ===
video_path = "/home/kevin-zhou/Videos/sample.mp4"
frame_rate = 5  # process every 5th frame

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Models ===
dpt_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device).eval()
dpt_processor = AutoProcessor.from_pretrained("Intel/dpt-hybrid-midas")

depth_anything_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=0 if torch.cuda.is_available() else -1)

seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512").to(device).eval()
seg_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
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

# === Helpers ===
def decode_segmap(segmap, colors):
    h, w = segmap.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(colors)):
        seg_rgb[segmap == i] = colors[i]
    return seg_rgb

def overlay_segmentation(image_pil, segmap, alpha=0.6):
    seg_rgb = decode_segmap(segmap, ADE20K_COLORS)
    seg_img = Image.fromarray(seg_rgb).resize(image_pil.size)
    return Image.blend(image_pil.convert("RGB"), seg_img.convert("RGB"), alpha)

# === Main Video Processing ===
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_rate == 0:
        image_pil = ImageOps.exif_transpose(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        # Segmentation
        seg_inputs = seg_extractor(images=image_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            seg_outputs = seg_model(**seg_inputs)
            seg_pred = torch.argmax(seg_outputs.logits, dim=1).squeeze().cpu().numpy()

        # DPT Depth
        dpt_inputs = dpt_processor(images=image_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            dpt_depth = dpt_model(**dpt_inputs).predicted_depth[0].squeeze().cpu().numpy()
        dpt_depth_vis = (dpt_depth - dpt_depth.min()) / (dpt_depth.max() - dpt_depth.min())

        # Depth Anything
        da_result = depth_anything_pipe(image_pil)
        da_depth = np.array(da_result["depth"])
        da_depth_vis = (da_depth - da_depth.min()) / (da_depth.max() - da_depth.min())

        # Prepare visuals
        image_np = np.array(image_pil)
        overlay_img = np.array(overlay_segmentation(image_pil, seg_pred))
        dpt_color = cv2.applyColorMap((dpt_depth_vis * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        da_color = cv2.applyColorMap((da_depth_vis * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)

        # Resize all to the same height for horizontal stacking
        target_h = image_np.shape[0]
        def resize_to_height(img):
            scale = target_h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), target_h))

        images_combined = np.hstack([
            resize_to_height(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)),
            resize_to_height(cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)),
            resize_to_height(dpt_color),
            resize_to_height(da_color)
        ])

        cv2.imshow("Combined Output [Original | Seg | DPT | DA]", images_combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
