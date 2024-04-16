import kornia
import torch.nn as nn
import torch


class SpatialConsistencyLoss(nn.Module):
    def __init__(self, left_to_right_transform, left_intrinsics, right_intrinsics):
        super().__init__()

        self.ssim_loss = kornia.losses.SSIMLoss(11)
        self.l1_loss = nn.L1Loss()
        self.left_to_right_transform = left_to_right_transform
        self.left_intrinsics = left_intrinsics
        self.right_intrinsics = right_intrinsics
        self.weight = 0.9

    def forward(self, left, right, left_depth, right_depth):
        gen_right = kornia.geometry.warp_frame_depth(left,
                                                     right_depth,
                                                     torch.inverse(self.left_to_right_transform),
                                                     self.left_intrinsics)

        gen_left = kornia.geometry.warp_frame_depth(right,
                                                    left_depth,
                                                    self.left_to_right_transform,
                                                    self.right_intrinsics)

        right_loss = (self.weight * self.ssim_loss(gen_right, right)
                      + (1 - self.weight) * self.l1_loss(gen_right, right))

        left_loss = (self.weight * self.ssim_loss(gen_left, left)
                     + (1 - self.weight) * self.l1_loss(gen_left, left))

        return (left_loss + right_loss) / 2
