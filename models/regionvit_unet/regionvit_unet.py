import torch.nn as nn

from models.regionvit_unet.regionvit import regionvit_tiny_224

class RegionViT_UNET(nn.Module):
    def __init__(self) -> None:

        super().__init__()

        self.normalize = True
        self.regionvit = regionvit_tiny_224()

    def forward(self, x_in):
        logits = self.regionvit(x_in)
        return logits