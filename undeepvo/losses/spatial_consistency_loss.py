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
        self.weight = 0.85

    def forward(self, left, right, left_depth, right_depth):
        batch_size = left.shape[0]

        gen_right = kornia.geometry.warp_frame_depth(left,
                                                     right_depth,
                                                     torch.inverse(self.left_to_right_transform[0:batch_size]),
                                                     self.left_intrinsics[0:batch_size])

        gen_left = kornia.geometry.warp_frame_depth(right,
                                                    left_depth,
                                                    self.left_to_right_transform[0:batch_size],
                                                    self.right_intrinsics[0:batch_size])

        right_loss = (self.weight * self.ssim_loss(gen_right, right)
                      + (1 - self.weight) * self.l1_loss(gen_right, right))

        left_loss = (self.weight * self.ssim_loss(gen_left, left)
                     + (1 - self.weight) * self.l1_loss(gen_left, left))

        return (left_loss + right_loss) / 2
