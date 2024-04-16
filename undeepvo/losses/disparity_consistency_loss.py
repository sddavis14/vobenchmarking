import kornia
import torch.nn as nn
import torch
from .helpers import generate_disparity_from_depth


class DisparityConsistencyLoss(nn.Module):
    def __init__(self, baseline, focal_length, left_to_right_transform, left_intrinsics, right_intrinsics):
        super().__init__()

        self.l1_loss = nn.L1Loss()
        self.left_to_right_transform = left_to_right_transform
        self.left_intrinsics = left_intrinsics
        self.right_intrinsics = right_intrinsics
        self.weight = 0.9

        self.baseline = baseline
        self.focal_length = focal_length

    def forward(self, left_depth, right_depth):
        left_disparity = generate_disparity_from_depth(left_depth, self.baseline, self.focal_length)
        right_disparity = generate_disparity_from_depth(right_depth, self.baseline, self.focal_length)

        gen_right_depth = kornia.geometry.warp_frame_depth(left_disparity,
                                                           right_depth,
                                                           torch.inverse(self.left_to_right_transform),
                                                           self.left_intrinsics)

        gen_left_depth = kornia.geometry.warp_frame_depth(right_disparity,
                                                          left_depth,
                                                          self.left_to_right_transform,
                                                          self.right_intrinsics)

        right_loss = self.weight * self.l1_loss(gen_right_depth, right_depth)

        left_loss = self.weight * self.l1_loss(gen_left_depth, left_depth)

        return (left_loss + right_loss) / 2
