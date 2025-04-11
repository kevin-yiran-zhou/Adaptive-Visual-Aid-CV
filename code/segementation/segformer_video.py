import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import cv2
import numpy as np
import time

# Load model
device = torch.device("cpu")
print("Using device:", device)

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
id2label = model.config.id2label

# ADE20K color map
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

def overlay_segmentation(image, segmap, alpha=0.6):
    seg_rgb = decode_segmap(segmap)
    seg_img = Image.fromarray(seg_rgb).resize(image.size)
    return Image.blend(image.convert("RGB"), seg_img.convert("RGB"), alpha)

# --- Video setup ---
video_path = "/home/kevin-zhou/Desktop/UMich/WeilandLab/segmentation/images/aa.mp4"
cap = cv2.VideoCapture(video_path)

original_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Original video FPS: {original_fps}")
selected_fps = 1
print(f"Selected processing FPS: {selected_fps}")

frame_interval = int(original_fps / selected_fps) if original_fps >= selected_fps else 1
frame_count = 0

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    # Convert frame to PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    overlay = overlay_segmentation(image, pred)
    result = np.array(overlay)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # --- Draw Legend ---
    unique_classes = np.unique(pred)
    start_y = 30
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    for cls_id in unique_classes:
        if cls_id < len(id2label):
            label = id2label[cls_id]
            color = [int(c) for c in ADE20K_COLORS[cls_id]]
            y_pos = start_y + 20 * int(np.where(unique_classes == cls_id)[0][0])
            
            # Draw color box
            cv2.rectangle(result, (10, y_pos - 12), (30, y_pos + 5), color, -1)
            # Draw label text
            cv2.putText(result, label, (35, y_pos), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Show result with legend
    cv2.imshow("SegFormer Video Overlay", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    outputs = model(**inputs)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s | FPS: {1/inference_time:.2f}")

cap.release()
cv2.destroyAllWindows()