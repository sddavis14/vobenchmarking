import kornia
import torch.nn as nn
import torch
from .helpers import generate_transformation


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, intrinsics):
        super().__init__()

        self.ssim_loss = kornia.losses.SSIMLoss(11)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.intrinsics = intrinsics
        self.weight = 0.85

    def forward(self, current_image, next_image,
                current_depth, next_depth,
                current_translation, current_rotation,
                next_translation, next_rotation):
        next_to_current = generate_transformation(current_translation, current_rotation)
        current_to_next = generate_transformation(next_translation, next_rotation)

        batch_size = current_image.shape[0]

        gen_next_image = kornia.geometry.warp_frame_depth(current_image,
                                                          next_depth,
                                                          next_to_current,
                                                          self.intrinsics[0:batch_size])

        gen_current_image = kornia.geometry.warp_frame_depth(next_image,
                                                             current_depth,
                                                             current_to_next,
                                                             self.intrinsics[0:batch_size])

        prev_loss = (self.weight * self.ssim_loss(gen_next_image, next_image)
                     + (1 - self.weight) * self.l1_loss(gen_next_image, next_image))

        current_loss = (self.weight * self.ssim_loss(gen_current_image, current_image)
                        + (1 - self.weight) * self.l1_loss(gen_current_image, current_image))

        return ((prev_loss + current_loss) / 2)
