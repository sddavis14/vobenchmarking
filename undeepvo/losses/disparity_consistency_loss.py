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
        self.weight = 1e-6

        self.baseline = baseline
        self.focal_length = focal_length

    def forward(self, left_depth, right_depth):
        batch_size = left_depth.shape[0]
        image_width = left_depth * left_depth.shape[3]

        left_disparity = image_width * generate_disparity_from_depth(left_depth, self.baseline, self.focal_length)
        right_disparity = image_width * generate_disparity_from_depth(right_depth, self.baseline, self.focal_length)

        gen_right_disparity = kornia.geometry.warp_frame_depth(left_disparity,
                                                           right_depth,
                                                           torch.inverse(self.left_to_right_transform[0:batch_size]),
                                                           self.left_intrinsics[0:batch_size])

        gen_left_disparity = kornia.geometry.warp_frame_depth(right_disparity,
                                                          left_depth,
                                                          self.left_to_right_transform[0:batch_size],
                                                          self.right_intrinsics[0:batch_size])

        right_loss = self.weight * self.l1_loss(gen_right_disparity, right_disparity)

        left_loss = self.weight * self.l1_loss(gen_left_disparity, left_disparity)

        return (left_loss + right_loss) / 2
