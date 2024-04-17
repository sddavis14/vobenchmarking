import kornia
import torch.nn as nn
import torch
from .helpers import generate_transformation

class PoseConsistencyLoss(nn.Module):
    def __init__(self, extrinsics_left_to_right):
        super().__init__()

        self.l1_loss = nn.L1Loss()
        self.left_to_right = extrinsics_left_to_right

        self.translation_weight = 0.01
        self.rotation_weight = 0.1

        self.transform_weight = 0.01

    def forward(self, position_left, rotation_left, position_right, rotation_right):
        batch_size = position_left.shape[0]

        left_to_right_current = self.left_to_right[0:batch_size]

        left_next_to_current = generate_transformation(position_left, rotation_left)
        right_next_to_current = generate_transformation(position_right, rotation_right)

        left_next_to_right_next = torch.inverse(right_next_to_current) @ left_to_right_current @ left_next_to_current

        transform_loss = self.transform_weight * self.l1_loss(left_next_to_right_next, left_to_right_current)

        return transform_loss
