import torch
import torch.nn as nn

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
