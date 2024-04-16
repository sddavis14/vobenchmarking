import kornia
import torch.nn as nn
import torch


class PoseConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = nn.L1Loss()

        self.translation_weight = 1
        self.rotation_weight = 1

    def forward(self, position_1, position_2, rotation_1, rotation_2):

        translation_loss = self.translation_weight * self.l1_loss(position_1, position_2)
        rotation_loss = self.rotation_weight * self.l1_loss(rotation_1, rotation_2)

        return (translation_loss + rotation_loss) / 2.0
