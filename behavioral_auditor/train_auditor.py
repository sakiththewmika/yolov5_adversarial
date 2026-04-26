import os
import torch
from auditor import MultiScaleAuditor
from loss import calculate_mse_loss
from bh_utils import (
    setup_paths,
    load_yolo_model_and_attach_hooks,
    load_images_from_folder,
)

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
ROOT, RUN_DIR = setup_paths()
WEIGHTS_DIR = RUN_DIR / 'saved_model'
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Define data and model paths
CLEAN_DATA_DIR = ROOT / 'data'/ 'bdd_subset_2' / 'train' / 'images'
YOLO_MODEL_WEIGHTS = ROOT / 'runs' / 'train' / 's_coco_e300_4Class_Vehicle' / 'weights' / 'best.pt'

features = {}
yolo = load_yolo_model_and_attach_hooks(ROOT, YOLO_MODEL_WEIGHTS, features)

# ---------------------------------------------------------
# 2. TRAINING LOOP
# ---------------------------------------------------------
def train_auditor(epochs=10):
    auditor = MultiScaleAuditor().cuda()
    optimizer = torch.optim.Adam(auditor.parameters(), lr=0.001)

    print(f"Loading clean training data from {CLEAN_DATA_DIR}...")
    clean_data = load_images_from_folder(CLEAN_DATA_DIR)
    
    yolo.eval() 
    auditor.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for _, img_tensor in clean_data:
            with torch.no_grad():
                _ = yolo(img_tensor) 
                
            feat_L0 = features['L0'].clone().detach().requires_grad_(True)
            feat_L1 = features['L1'].clone().detach().requires_grad_(True)
            feat_L3 = features['L3'].clone().detach().requires_grad_(True)
            actual_feat_L5 = features['L5'].clone().detach()
            
            optimizer.zero_grad()
            
            predicted_feat_L5 = auditor(feat_L0, feat_L1, feat_L3)
            
            loss = calculate_mse_loss(predicted_feat_L5, actual_feat_L5)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss / len(clean_data):.5f}")
    
    save_path = WEIGHTS_DIR / f'auditor_latest_{epochs}_epochs.pt'
    torch.save(auditor.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ---------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    train_auditor(epochs=10) 
    print(f"\nProcess Complete! Check the '{RUN_DIR}' folder.")
