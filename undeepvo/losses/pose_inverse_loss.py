import kornia
import torch.nn as nn
import torch
from .helpers import generate_transformation


class PoseInverseLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.weight = 0.01

    def forward(self, 
                current_translation, current_rotation,
                next_translation, next_rotation):
        next_to_current = generate_transformation(current_translation, current_rotation)
        current_to_next = generate_transformation(next_translation, next_rotation)

        gen_current_to_next = torch.inverse(next_to_current)
        
        loss = self.weight * self.l1_loss(gen_current_to_next, current_to_next)

        return loss
