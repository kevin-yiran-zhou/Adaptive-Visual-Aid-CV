import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import time

# ----- Device -----
device = torch.device("cpu")
print(f"Using device: {device}")

# ----- Load TorchVision's pretrained DeepLabV3 -----
weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet101(weights=weights).to(device)
model.eval()

# ----- Preprocessing -----
preprocess = weights.transforms()

# ----- VOC Class Names & Color Map -----
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'potted plant', 'sheep',
    'sofa', 'train', 'tv/monitor'
]

VOC_COLORS = np.array([
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
])

def decode_segmap(segmentation):
    r = np.zeros_like(segmentation).astype(np.uint8)
    g = np.zeros_like(segmentation).astype(np.uint8)
    b = np.zeros_like(segmentation).astype(np.uint8)
    for l in range(len(VOC_COLORS)):
        idx = segmentation == l
        r[idx] = VOC_COLORS[l, 0]
        g[idx] = VOC_COLORS[l, 1]
        b[idx] = VOC_COLORS[l, 2]
    return np.stack([r, g, b], axis=2)

def overlay_segmentation(image_pil, segmap, alpha=0.6):
    seg_rgb = decode_segmap(segmap)
    seg_img = Image.fromarray(seg_rgb).resize(image_pil.size)
    return Image.blend(image_pil.convert("RGB"), seg_img.convert("RGB"), alpha)

def draw_legend(segmap):
    unique_classes = np.unique(segmap)
    legend_patches = []
    for cls_id in unique_classes:
        if cls_id < len(VOC_CLASSES):
            name = VOC_CLASSES[cls_id]
            color = VOC_COLORS[cls_id] / 255
            patch = plt.Rectangle((0, 0), 1, 1, color=color, label=name)
            legend_patches.append(patch)
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

# ----- Inference on all PNG images -----
image_files = sorted(glob.glob("/home/kevin-zhou/Desktop/UMich/WeilandLab/Adaptive-Visual-Aid-CV/images/*.jpeg"))

for img_path in image_files:
    print(f"Processing {img_path}...")
    start_time = time.time()
    image = Image.open(img_path).convert("RGB")
    ### resize image ###
    resize_factor = 1
    new_size = (int(image.size[0] // resize_factor), int(image.size[1] // resize_factor))
    image = image.resize(new_size, Image.BILINEAR)
    ####################
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"]
        prediction = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s | FPS: {1/inference_time:.2f}")
    overlay = overlay_segmentation(image, prediction)

    plt.figure(figsize=(10, 5))
    plt.imshow(overlay)
    plt.title("Overlayed Segmentation")
    plt.axis("off")
    draw_legend(prediction)
    plt.tight_layout()
    plt.show()