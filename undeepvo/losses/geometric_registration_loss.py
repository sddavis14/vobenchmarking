import kornia
import torch.nn as nn
import torch
from .helpers import generate_transformation


class GeometricRegistrationLoss(nn.Module):
    def __init__(self, intrinsics):
        super().__init__()

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.intrinsics = intrinsics
        self.weight = 0.9

    def forward(self, current_depth, next_depth,
                current_translation, current_rotation,
                next_translation, next_rotation):
        next_to_current = generate_transformation(current_translation, current_rotation)
        current_to_next = generate_transformation(next_translation, next_rotation)

        gen_next_depth = kornia.geometry.warp_frame_depth(current_depth,
                                                          next_depth,
                                                          next_to_current,
                                                          self.intrinsics)

        gen_current_depth = kornia.geometry.warp_frame_depth(next_depth,
                                                             current_depth,
                                                             current_to_next,
                                                             self.intrinsics)

        prev_loss = self.weight * self.l1_loss(gen_next_depth, next_depth)
        current_loss = self.weight * self.l1_loss(gen_current_depth, current_depth)

        return (prev_loss + current_loss) / 2
