import torch
import torch.nn as nn

def calculate_cosine_anomaly_score(predicted_features, actual_features):
    """
    Calculates the anomaly score based on cosine similarity between
    predicted and actual feature maps. It includes spatial smoothing.
    """
    # Use Cosine Distance + Spatial Smoothing to ignore background noise
    cos_sim = torch.nn.functional.cosine_similarity(predicted_features, actual_features, dim=1)
    
    # Insert a single dummy channel dimension at dim=1. Shape becomes [Batch, 1, H, W]
    spatial_distance = (1.0 - cos_sim).unsqueeze(1) 
    
    # Apply 3x3 Average Pool to isolate dense patches and smooth background noise
    smoothed_anomaly = torch.nn.functional.avg_pool2d(spatial_distance, kernel_size=3, stride=1, padding=1)
    
    return smoothed_anomaly

def calculate_mse_loss(predicted_features, actual_features):
    """
    Calculates the Mean Squared Error loss between predicted and actual features.
    """
    loss_fn = nn.MSELoss()
    return loss_fn(predicted_features, actual_features)
