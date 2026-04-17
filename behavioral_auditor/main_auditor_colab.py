import os
import sys
import glob
import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. PATH SETUP & OUTPUT DIRECTORIES
# ---------------------------------------------------------
# Add the root yolov5_adversarial folder to sys.path so we can load local modules
ROOT = Path('/content/drive/MyDrive/Research/test_adv/yolov5_adversarial')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Generate a timestamp for this specific run (e.g., 20260328_201530)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define where to save outputs dynamically based on the timestamp
OUTPUT_DIR = Path('/content/drive/MyDrive/Research/test_adv/yolov5_adversarial/behavioral_auditor/outputs')
RUN_DIR = OUTPUT_DIR / timestamp

WEIGHTS_DIR = RUN_DIR / 'saved_model'
CLEAN_HEATMAP_DIR = RUN_DIR / 'clean_heatmaps'
PATCHED_HEATMAP_DIR = RUN_DIR / 'patched_heatmaps'

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CLEAN_HEATMAP_DIR, exist_ok=True)
os.makedirs(PATCHED_HEATMAP_DIR, exist_ok=True)

print(f"Creating output directories under: {RUN_DIR}")

# Define your data paths 
CLEAN_DATA_DIR = '/content/drive/MyDrive/Research/test_adv/yolov5_adversarial/runs/test_adversarial/20260329-100705_bdd_patch_car_2/clean/images'
PATCHED_DATA_DIR = '/content/drive/MyDrive/Research/test_adv/yolov5_adversarial/runs/test_adversarial/20260329-100705_bdd_patch_car_2/proper_patched/images'

# ---------------------------------------------------------
# 2. YOLOv5 HOOKS & MODEL LOADING
# ---------------------------------------------------------
features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

print("Loading local YOLOv5 model...")
model_path = '/content/drive/MyDrive/Research/test_adv/yolov5_adversarial/runs/train/s_coco_e300_4Class_Vehicle/weights/best.pt' 
yolo = torch.hub.load(str(ROOT), 'custom', path=str(model_path), source='local')

sequential_layers = None
for m in yolo.modules():
    if isinstance(m, nn.Sequential) and len(list(m.children())) > 15:
        sequential_layers = m
        break

if sequential_layers is None:
    raise RuntimeError("Could not locate the YOLOv5 nn.Sequential layer list!")

# Hooking Layers 0, 1, 3 (Inputs) and Layer 5 (Target)
layer_idxs = {'L0': 0, 'L1': 1, 'L3': 3, 'L5': 5}
print(f"Attaching hooks to Layers: {list(layer_idxs.values())}...")

for name, idx in layer_idxs.items():
    sequential_layers[idx].register_forward_hook(get_features(name))

# --------------------------------------------
# 3. MULTI-SCALE AUDITOR NETWORK (For YOLOv5s)
# --------------------------------------------
class MultiScaleAuditor(nn.Module):
    def __init__(self):
        super(MultiScaleAuditor, self).__init__()
        
        # Layer 0: Actual is 32 channels, 320x320. We need 40x40. (Stride 8)
        self.process_L0 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=8, stride=8),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
        # Layer 1: Actual is 64 channels, 160x160. We need 40x40. (Stride 4)
        self.process_L1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
        # Layer 3: Actual is 128 channels, 80x80. We need 40x40. (Stride 2)
        self.process_L3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
        # Total channels after concat: 128 + 128 + 128 = 384
        # Target Layer 5 has 256 channels in the YOLOv5s model
        self.regressor = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            # Output exactly 256 channels so it mathematically aligns with Layer 5
            nn.Conv2d(256, 256, kernel_size=1) 
        )

    def forward(self, l0, l1, l3):
        # Mathematically align all spatial dimensions to 40x40
        x0 = self.process_L0(l0)
        x1 = self.process_L1(l1)
        x3 = self.process_L3(l3)
        
        # Concatenate along the channel dimension
        fused = torch.cat((x0, x1, x3), dim=1)
        
        # Predict Layer 5
        return self.regressor(fused)

auditor = MultiScaleAuditor().cuda()
optimizer = torch.optim.Adam(auditor.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ---------------------------------------------------------
# 4. SIMPLE DATA LOADER & TRANSFORM
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

def load_images_from_folder(folder_path):
    images = []
    for img_path in glob.glob(os.path.join(str(folder_path), '*.jpg')):
        img = Image.open(img_path).convert('RGB')
        images.append((img_path, transform(img).unsqueeze(0).cuda()))
    return images

# ---------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------
def train_auditor(epochs=10):
    print(f"Loading clean training data from {CLEAN_DATA_DIR}...")
    clean_data = load_images_from_folder(CLEAN_DATA_DIR)
    
    yolo.eval() 
    auditor.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for _, img_tensor in clean_data:
            with torch.no_grad():
                _ = yolo(img_tensor) 
                
            # Grab inputs and target: clone and detach to break the inference lock
            # requires_grad_(True) tells PyTorch "this is the start of a new computation graph"
            feat_L0 = features['L0'].clone().detach().requires_grad_(True)
            feat_L1 = features['L1'].clone().detach().requires_grad_(True)
            feat_L3 = features['L3'].clone().detach().requires_grad_(True)
            
            # The target does not need gradients, it just needs to be detached
            actual_feat_L5 = features['L5'].clone().detach()
            
            optimizer.zero_grad()
            
            # Forward pass with multiple scales
            predicted_feat_L5 = auditor(feat_L0, feat_L1, feat_L3)
            
            loss = criterion(predicted_feat_L5, actual_feat_L5)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss / len(clean_data):.5f}")
    
    save_path = WEIGHTS_DIR / f'auditor_latest_{epochs}_epochs.pt'
    torch.save(auditor.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ---------------------------------------------------------
# 6. TESTING & SAVING HEATMAPS
# ---------------------------------------------------------
def save_heatmap(img_name, heatmap_data, save_prefix):
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Smoothed Anomaly Score')
    plt.title(f"Internal Anomaly Heatmap: {save_prefix}")
    
    target_dir = CLEAN_HEATMAP_DIR if save_prefix == 'clean' else PATCHED_HEATMAP_DIR
    save_file = target_dir / f"{save_prefix}_{os.path.basename(img_name)}.png"
    
    plt.savefig(save_file)
    plt.close()
    print(f"Saved heatmap to {save_file}")

def test_auditor():
    print("\n--- Running Feasibility Test ---")
    auditor.eval()
    yolo.eval()
    
    clean_test_data = load_images_from_folder(CLEAN_DATA_DIR)[:5] 
    patched_test_data = load_images_from_folder(PATCHED_DATA_DIR)[:5]
    
    all_test_data = [("clean", clean_test_data), ("patched", patched_test_data)]
    
    for category, data in all_test_data:
        for img_path, img_tensor in data:
            with torch.no_grad():
                _ = yolo(img_tensor)
                
                feat_L0 = features['L0']
                feat_L1 = features['L1']
                feat_L3 = features['L3']
                actual_feat_L5 = features['L5']
                
                predicted_feat_L5 = auditor(feat_L0, feat_L1, feat_L3)
                
                # Use Cosine Distance + Spatial Smoothing to ignore background noise
                cos_sim = torch.nn.functional.cosine_similarity(predicted_feat_L5, actual_feat_L5, dim=1)
                
                # FIX: Insert a single dummy channel dimension at dim=1. Shape becomes [1, 1, 40, 40]
                spatial_distance = (1.0 - cos_sim).unsqueeze(1) 
                
                # Apply 3x3 Average Pool to isolate the dense patch and ignore 1-pixel tree spikes
                smoothed_anomaly = torch.nn.functional.avg_pool2d(spatial_distance, kernel_size=3, stride=1, padding=1)
                
                # Squeeze out the batch and channel dims to get a flat 2D [40, 40] heatmap for matplotlib
                smoothed_anomaly = smoothed_anomaly.squeeze().cpu().numpy()
                
                save_heatmap(img_path, smoothed_anomaly, category)

# ---------------------------------------------------------
# 7. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    train_auditor(epochs=50) 
    test_auditor()
    print(f"\nProcess Complete! Check the '{RUN_DIR}' folder.")