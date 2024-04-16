#!/usr/bin/env python3

import warnings ; warnings.warn = lambda *args,**kwargs: None

import torch
import kornia

from model.full_model import UnDeepVO
from data_processing.four_seasons import FourSeasonsDataset
from matplotlib import pyplot as plt

from losses.pose_consistency_loss import PoseConsistencyLoss
from losses.temporal_consistency_loss import TemporalConsistencyLoss
from losses.spatial_consistency_loss import SpatialConsistencyLoss
from losses.geometric_registration_loss import GeometricRegistrationLoss
from losses.disparity_consistency_loss import DisparityConsistencyLoss
import numpy as np


def run():
    fig0 = plt.figure()
    display_img = plt.imshow(np.zeros(shape=(128, 256)), interpolation='nearest', cmap='gray', vmin=-1, vmax=1)
    fig1 = plt.figure()
    depth_img = plt.imshow(np.zeros(shape=(128, 256)), interpolation='nearest', cmap='plasma', vmin=1, vmax=10)

    device = torch.device("mps")

    model = UnDeepVO()
    model.to(device)

    dataset = FourSeasonsDataset()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 18

    right_intrinsics_tensor = torch.from_numpy(
        np.repeat(np.expand_dims(dataset.right_intrinsics(), axis=0), batch_size, axis=0)).to(device)
    left_intrinsics_tensor = torch.from_numpy(
        np.repeat(np.expand_dims(dataset.right_intrinsics(), axis=0), batch_size, axis=0)).to(device)
    extrinsics_tensor = torch.from_numpy(
        np.repeat(np.expand_dims(dataset.left_to_right_camera_extrinsics(), axis=0), batch_size, axis=0)).to(device)

    pose_loss = PoseConsistencyLoss()
    right_temporal_consistency_loss = TemporalConsistencyLoss(right_intrinsics_tensor)
    left_temporal_consistency_loss = TemporalConsistencyLoss(left_intrinsics_tensor)
    spatial_consistency_loss = SpatialConsistencyLoss(extrinsics_tensor,
                                                      left_intrinsics_tensor,
                                                      right_intrinsics_tensor)
    disparity_consistency_loss = DisparityConsistencyLoss(dataset.baseline(),
                                                          dataset.focal_length(),
                                                          extrinsics_tensor,
                                                          left_intrinsics_tensor,
                                                          right_intrinsics_tensor)

    left_geometric_registration_loss = GeometricRegistrationLoss(left_intrinsics_tensor)
    right_geometric_registration_loss = GeometricRegistrationLoss(right_intrinsics_tensor)

    for left, right, left_next, right_next in dataset.training_batch_generator(batch_size):
        optimizer.zero_grad()

        left_tensor = torch.from_numpy(left).to(device)
        right_tensor = torch.from_numpy(right).to(device)
        left_next_tensor = torch.from_numpy(left_next).to(device)
        right_next_tensor = torch.from_numpy(right_next).to(device)

        left_depth, (left_rotation, left_translation) = model(left_tensor, left_next_tensor)
        right_depth, (right_rotation, right_translation) = model(right_tensor, right_next_tensor)

        left_next_depth, (left_next_rotation, left_next_translation) = model(left_next_tensor, left_tensor)
        right_next_depth, (right_next_rotation, right_next_translation) = model(right_next_tensor, right_tensor)

        pose_loss_1_val = pose_loss(left_translation, right_translation, left_rotation, right_rotation)
        pose_loss_2_val = pose_loss(left_next_translation, right_next_translation, left_next_rotation,
                                    right_next_rotation)

        spatial_consistency_loss_1_val = spatial_consistency_loss(left_tensor, right_tensor, left_depth, right_depth)
        spatial_consistency_loss_2_val = spatial_consistency_loss(left_next_tensor, right_next_tensor, left_next_depth,
                                                                  right_next_depth)

        left_temporal_consistency_loss_val = left_temporal_consistency_loss(left_tensor, left_next_tensor, left_depth,
                                                                            left_next_depth,
                                                                            left_translation, left_rotation,
                                                                            left_next_translation, left_rotation)

        right_temporal_consistency_loss_val = right_temporal_consistency_loss(right_tensor, right_next_tensor,
                                                                              right_depth, right_next_depth,
                                                                              right_translation, right_rotation,
                                                                              right_next_translation, right_rotation)

        disparity_consistency_loss_1_val = disparity_consistency_loss(left_depth, right_depth)
        disparity_consistency_loss_2_val = disparity_consistency_loss(left_next_depth, right_next_depth)

        geometric_registration_loss_left_val = left_geometric_registration_loss(left_depth, left_next_depth,
                                                                                left_translation, left_rotation,
                                                                                left_next_translation,
                                                                                left_next_rotation)

        geometric_registration_loss_right_val = right_geometric_registration_loss(left_depth, left_next_depth,
                                                                                  left_translation, left_rotation,
                                                                                  left_next_translation,
                                                                                  left_next_rotation)

        loss = pose_loss_1_val + pose_loss_2_val \
               + spatial_consistency_loss_1_val + spatial_consistency_loss_2_val \
               + left_temporal_consistency_loss_val + right_temporal_consistency_loss_val \
               + disparity_consistency_loss_1_val + disparity_consistency_loss_2_val \
               + geometric_registration_loss_left_val + geometric_registration_loss_right_val

        loss.backward()
        optimizer.step()

        display_img.set_data(left_tensor.cpu().detach().numpy()[0, 0])
        depth_img.set_data(left_depth.cpu().detach().numpy()[0, 0])
        fig0.canvas.draw()
        fig1.canvas.draw()
        plt.pause(0.01)

        print(loss.item())


if __name__ == "__main__":
    run()
