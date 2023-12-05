import torch
import torch.nn as nn
import monai

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        # y_true = torch.squeeze(y_true, dim=1).long()
        return self.loss(y_pred, y_true)


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(sigmoid = True)
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        cross_entropy = self.cross_entropy(y_pred, y_true)
        return dice + cross_entropy