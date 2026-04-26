import os
import sys
import glob
import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

def setup_paths():
    """
    Adds the root directory to sys.path and creates a timestamped run directory.
    Returns the root path and the run directory path.
    """
    root_path = Path.cwd()
    sys.path.append(str(root_path))
    run_dir = root_path / 'runs' / 'auditor' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(run_dir, exist_ok=True)
    print(f'Creating output directories under: {run_dir}')
    return root_path, run_dir

def get_features_hook(features_dict):
    """
    Returns a hook function that captures the output of a layer.
    """
    def get_features(name):
        def hook(model, input, output):
            features_dict[name] = output.detach()
        return hook
    return get_features

def load_yolo_model_and_attach_hooks(root_path, model_weights_path, features_dict):
    """
    Loads the YOLOv5 model and attaches forward hooks to specified layers.
    """
    print("Loading local YOLOv5 model...")
    yolo = torch.hub.load(str(root_path), 'custom', path=str(model_weights_path), source='local')

    sequential_layers = None
    for m in yolo.modules():
        if isinstance(m, nn.Sequential) and len(list(m.children())) > 15:
            sequential_layers = m
            break

    if sequential_layers is None:
        raise RuntimeError("Could not locate the YOLOv5 nn.Sequential layer list!")

    layer_idxs = {'L0': 0, 'L1': 1, 'L3': 3, 'L5': 5}
    print(f"Attaching hooks to Layers: {list(layer_idxs.values())}...")
    
    get_features = get_features_hook(features_dict)
    for name, idx in layer_idxs.items():
        sequential_layers[idx].register_forward_hook(get_features(name))
    
    return yolo

def load_images_from_folder(folder_path):
    """
    Loads images from a folder and applies the necessary transformations.
    """
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    
    images = []
    for img_path in glob.glob(os.path.join(str(folder_path), '*.jpg')):
        img = Image.open(img_path).convert('RGB')
        images.append((img_path, transform(img).unsqueeze(0).cuda()))
    return images

def draw_status_on_image(image, attack_detected, max_anomaly):
    """
    Draws the security status (secure or attack detected) on the top-left corner of the image.
    """
    # Dynamically scale font based on image width, but keep it readable
    font_scale = max(image.shape[1] / 1200, 0.6) 
    thickness = max(int(font_scale * 2), 1)
    
    if attack_detected:
        text = f"WARNING: ADVERSARIAL ATTACK (Anomaly: {max_anomaly:.2f})"
        text_color = (0, 0, 255)  # Red
        bg_color = (0, 0, 0)      # Black
        # Add red border
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), text_color, 8)
    else:
        text = f"SYSTEM SECURE (Anomaly: {max_anomaly:.2f})"
        text_color = (0, 255, 0)  # Green
        bg_color = (0, 0, 0)      # Black
    
    # Get text size to draw a solid background box
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Safe coordinates (pushed down slightly so it doesn't crop)
    start_x = 20
    start_y = text_height + 30 
    
    # Draw solid black background rectangle for readability
    cv2.rectangle(image, 
                  (start_x - 5, start_y - text_height - 5), 
                  (start_x + text_width + 5, start_y + baseline + 5), 
                  bg_color, cv2.FILLED)
    
    # Draw the text over the background
    cv2.putText(image, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    return image
