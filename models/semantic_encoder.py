import torch.nn as nn


class SemanticEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
