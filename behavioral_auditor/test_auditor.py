import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from auditor import MultiScaleAuditor
from bh_utils import (
    load_yolo_model_and_attach_hooks,
    load_images_from_folder,
)
from loss import calculate_cosine_anomaly_score


def _find_latest_auditor_weights(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob('*/saved_model/*.pt'), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f'No auditor checkpoint found under {run_dir}')
    return candidates[0]

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / 'runs' / 'auditor' 
LATEST_RUN = sorted(os.listdir(RUN_DIR))[-1]
WEIGHTS_DIR = RUN_DIR / LATEST_RUN / 'saved_model'
AUDITOR_WEIGHTS = _find_latest_auditor_weights(RUN_DIR)

# Define data and model paths
CLEAN_DATA_DIR = ROOT / 'runs' / 'test_adversarial' / '20260322-215648_base' / 'clean' / 'images'
ADVERSARIAL_DATA_DIR = ROOT / 'runs' / 'test_adversarial' / '20260322-215648_base' / 'proper_patched' / 'images'
YOLO_MODEL_WEIGHTS = ROOT / 'runs' / 'train' / 's_coco_e300_4Class_Vehicle' / 'weights' / 'best.pt'

# Define feature extraction layers
features = {}
AUDITOR_LAYERS = [0, 1, 3]
PREDICTION_LAYER = 5

# Load YOLOv5 model and attach hooks
print("Loading local YOLOv5 model...")
yolo = load_yolo_model_and_attach_hooks(ROOT, YOLO_MODEL_WEIGHTS, features)

# Load Auditor model
print("Loading trained auditor model...")
auditor = MultiScaleAuditor()
auditor.load_state_dict(torch.load(AUDITOR_WEIGHTS, map_location='cpu'))

# Load test data
print(f"Loading clean test data from {CLEAN_DATA_DIR}...")
clean_images = load_images_from_folder(CLEAN_DATA_DIR)
print(f"Loading adversarial test data from {ADVERSARIAL_DATA_DIR}...")
adv_images = load_images_from_folder(ADVERSARIAL_DATA_DIR)

auditor_device = clean_images[0][1].device if clean_images else (adv_images[0][1].device if adv_images else torch.device('cpu'))
auditor = auditor.to(auditor_device)
auditor.eval()

# ---------------------------------------------------------
# 2. INFERENCE & EVALUATION
# ---------------------------------------------------------

def test_auditor(images, data_type='Clean'):
    print(f"--- Testing on {data_type} Data ---")
    total_loss = 0.0
    for i, (_, img_tensor) in enumerate(images):
        # YOLO inference to get feature maps
        with torch.no_grad():
            yolo(img_tensor)
        
        # Extract features
        feat_L0 = features['L0'].clone().detach()
        feat_L1 = features['L1'].clone().detach()
        feat_L3 = features['L3'].clone().detach()
        actual_feat_L5 = features['L5'].clone().detach()
        features.clear()

        # Auditor prediction
        with torch.no_grad():
            predicted_feat_L5 = auditor(feat_L0, feat_L1, feat_L3)

        # Calculate loss
        loss = calculate_cosine_anomaly_score(predicted_feat_L5, actual_feat_L5).mean()
        total_loss += loss.item()
        print(f"Image {i+1}/{len(images)}, Cosine Anomaly Score: {loss.item():.4f}")

    avg_loss = total_loss / len(images)
    print(f"Average Cosine Anomaly Score on {data_type} Data: {avg_loss:.4f}")
    return avg_loss

# Run tests
clean_loss = test_auditor(clean_images, data_type='Clean')
adv_loss = test_auditor(adv_images, data_type='Adversarial')

# ---------------------------------------------------------
# 3. SAVE RESULTS
# ---------------------------------------------------------
results_file = RUN_DIR / LATEST_RUN / 'test_results.txt'
with open(results_file, 'w') as f:
    f.write(f"Average Cosine Anomaly Score on Clean Data: {clean_loss:.4f}\n")
    f.write(f"Average Cosine Anomaly Score on Adversarial Data: {adv_loss:.4f}\n")

print(f"Test results saved to {results_file}")

# Optional: Visualize results for one image
if len(clean_images) > 0 and len(adv_images) > 0:
    # Clean image
    yolo(clean_images[0][1])
    feat_L0_clean = features['L0'].clone().detach()
    feat_L1_clean = features['L1'].clone().detach()
    feat_L3_clean = features['L3'].clone().detach()
    actual_L5_clean = features['L5'].clone().detach()
    features.clear()
    predicted_L5_clean = auditor(feat_L0_clean, feat_L1_clean, feat_L3_clean)

    # Adversarial image
    yolo(adv_images[0][1])
    feat_L0_adv = features['L0'].clone().detach()
    feat_L1_adv = features['L1'].clone().detach()
    feat_L3_adv = features['L3'].clone().detach()
    actual_L5_adv = features['L5'].clone().detach()
    features.clear()
    predicted_L5_adv = auditor(feat_L0_adv, feat_L1_adv, feat_L3_adv)

    # Visualize
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(actual_L5_clean[0, 0].cpu().numpy(), cmap='viridis')
    axs[0, 0].set_title('Actual L5 (Clean)')
    axs[0, 1].imshow(predicted_L5_clean[0, 0].cpu().detach().numpy(), cmap='viridis')
    axs[0, 1].set_title('Predicted L5 (Clean)')
    axs[1, 0].imshow(actual_L5_adv[0, 0].cpu().numpy(), cmap='viridis')
    axs[1, 0].set_title('Actual L5 (Adv)')
    axs[1, 1].imshow(predicted_L5_adv[0, 0].cpu().detach().numpy(), cmap='viridis')
    axs[1, 1].set_title('Predicted L5 (Adv)')
    
    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(RUN_DIR / LATEST_RUN / 'feature_map_comparison.png')
    print(f"Feature map comparison saved to {RUN_DIR / LATEST_RUN / 'feature_map_comparison.png'}")

