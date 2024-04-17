#!/usr/bin/env python3

# suppress deprecation warnings in kornia
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
from losses.pose_inverse_loss import PoseInverseLoss

import numpy as np
from losses.helpers import generate_transformation
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import math

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(rank)
    
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(rank: int, world_size: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)

    gpu_id = rank

    if gpu_id == 0:
        sz = (192, 512)
        fig0 = plt.figure()
        plt.title('Left Next Img')
        display_img = plt.imshow(np.zeros(shape=sz), interpolation='nearest', cmap='gray', vmin=-1, vmax=1)
        
        fig3 = plt.figure()
        plt.title('Generated Left Img')
        display_img_gen = plt.imshow(np.zeros(shape=sz), interpolation='nearest', cmap='gray', vmin=-1, vmax=1)
        
        fig1 = plt.figure()
        plt.title('Left depth')
        depth_img = plt.imshow(np.zeros(shape=sz), interpolation='nearest', cmap='inferno', vmin=0, vmax=2)

        fig2 = plt.figure()
        plt.title('Right depth')
        depth_img_right = plt.imshow(np.zeros(shape=sz), interpolation='nearest', cmap='inferno', vmin=0, vmax=2)

        fig4 = plt.figure()
        plt.title('Generated Left Next Img')
        display_img_gen_next = plt.imshow(np.zeros(shape=sz), interpolation='nearest', cmap='gray', vmin=-1, vmax=1)
        
    device = torch.device(gpu_id)
    
    model = DDP(UnDeepVO().to(device), device_ids=[gpu_id], broadcast_buffers=False)

    model.load_state_dict(torch.load('weights_overnight_training'))

    dataset = FourSeasonsDataset()

    print('Dataset size: ' + str(len(dataset)))

    train_dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    right_intrinsics_tensor = torch.from_numpy(
        np.repeat(np.expand_dims(dataset.right_intrinsics(), axis=0), batch_size, axis=0)).to(device)
    left_intrinsics_tensor = torch.from_numpy(
        np.repeat(np.expand_dims(dataset.left_intrinsics(), axis=0), batch_size, axis=0)).to(device)
    extrinsics_tensor = torch.from_numpy(
        np.repeat(np.expand_dims(dataset.left_to_right_camera_extrinsics(), axis=0), batch_size, axis=0)).to(device)

    pose_loss = PoseConsistencyLoss(extrinsics_tensor).to(device)
    right_temporal_consistency_loss = TemporalConsistencyLoss(right_intrinsics_tensor).to(device)
    left_temporal_consistency_loss = TemporalConsistencyLoss(left_intrinsics_tensor).to(device)
    spatial_consistency_loss = SpatialConsistencyLoss(extrinsics_tensor,
                                                      left_intrinsics_tensor,
                                                      right_intrinsics_tensor).to(device)
    disparity_consistency_loss = DisparityConsistencyLoss(dataset.baseline(),
                                                          dataset.focal_length(),
                                                          extrinsics_tensor,
                                                          left_intrinsics_tensor,
                                                          right_intrinsics_tensor).to(device)

    left_geometric_registration_loss = GeometricRegistrationLoss(left_intrinsics_tensor).to(device)
    right_geometric_registration_loss = GeometricRegistrationLoss(right_intrinsics_tensor).to(device)

    left_pose_inverse_loss = PoseInverseLoss().to(device)
    right_pose_inverse_loss = PoseInverseLoss().to(device)

    for epoch in range(total_epochs):
        for step, (left_tensor_cpu, right_tensor_cpu, left_next_tensor_cpu, right_next_tensor_cpu) in enumerate(train_dataloader):
            optimizer.zero_grad()
            current_batch_size = left_tensor_cpu.shape[0]

            left_tensor = left_tensor_cpu.to(device)
            right_tensor = right_tensor_cpu.to(device)
            left_next_tensor = left_next_tensor_cpu.to(device)
            right_next_tensor = right_next_tensor_cpu.to(device)

            left_depth, (left_rotation, left_translation) = model(left_tensor, left_next_tensor)
            right_depth, (right_rotation, right_translation) = model(right_tensor, right_next_tensor)

            left_next_depth, (left_next_rotation, left_next_translation) = model(left_next_tensor, left_tensor)
            right_next_depth, (right_next_rotation, right_next_translation) = model(right_next_tensor, right_tensor)

            pose_loss_1_val = pose_loss(left_translation, right_translation, left_rotation, right_rotation)
            pose_loss_2_val = pose_loss(left_next_translation, right_next_translation, left_next_rotation,
                                        right_next_rotation)

            spatial_consistency_loss_1_val = spatial_consistency_loss(left_tensor, right_tensor, 
                                                                    left_depth, right_depth)
            spatial_consistency_loss_2_val = spatial_consistency_loss(left_next_tensor, right_next_tensor, 
                                                                    left_next_depth, right_next_depth)

            left_temporal_consistency_loss_val = left_temporal_consistency_loss(left_tensor, left_next_tensor, 
                                                                                left_depth, left_next_depth,
                                                                                left_translation, left_rotation,
                                                                                left_next_translation, left_next_rotation)

            right_temporal_consistency_loss_val = right_temporal_consistency_loss(right_tensor, right_next_tensor,
                                                                                right_depth, right_next_depth,
                                                                                right_translation, right_rotation,
                                                                                right_next_translation, right_next_rotation)

            disparity_consistency_loss_1_val = disparity_consistency_loss(left_depth, right_depth)
            disparity_consistency_loss_2_val = disparity_consistency_loss(left_next_depth, right_next_depth)

            geometric_registration_loss_left_val = left_geometric_registration_loss(left_depth, left_next_depth,
                                                                                    left_translation, left_rotation,
                                                                                    left_next_translation,
                                                                                    left_next_rotation)

            geometric_registration_loss_right_val = right_geometric_registration_loss(right_depth, right_next_depth,
                                                                                    right_translation, right_rotation,
                                                                                    right_next_translation,
                                                                                    right_next_rotation)

            left_pose_inverse_loss_val = left_pose_inverse_loss(left_translation, left_rotation, left_next_translation, left_next_rotation)
            right_pose_inverse_loss_val = right_pose_inverse_loss(right_translation, right_rotation, right_next_translation, right_next_rotation)

            loss = pose_loss_1_val + pose_loss_2_val \
                + spatial_consistency_loss_1_val + spatial_consistency_loss_2_val \
                + left_temporal_consistency_loss_val + right_temporal_consistency_loss_val \
                + disparity_consistency_loss_1_val + disparity_consistency_loss_2_val \
                + geometric_registration_loss_left_val + geometric_registration_loss_right_val \
                + left_pose_inverse_loss_val + right_pose_inverse_loss_val

            loss.backward()
            optimizer.step()
            
            if gpu_id == 0:
                gen_left_image = kornia.geometry.warp_frame_depth(right_tensor,
                                                left_depth,
                                                extrinsics_tensor[0:current_batch_size],
                                                right_intrinsics_tensor[0:current_batch_size])

                next_to_current = generate_transformation(left_translation, left_rotation)

                gen_left_next_image = kornia.geometry.warp_frame_depth(left_tensor,
                                                                left_next_depth,
                                                                next_to_current,
                                                                left_intrinsics_tensor[0:current_batch_size])

                display_img.set_data(left_next_tensor.cpu().detach().numpy()[0, 0])
                display_img_gen.set_data(gen_left_image.cpu().detach().numpy()[0, 0])
                display_img_gen_next.set_data(gen_left_next_image.cpu().detach().numpy()[0, 0])

                depth_img.set_data(np.log10(left_depth.cpu().detach().numpy()[0, 0]))
                depth_img_right.set_data(np.log10(right_depth.cpu().detach().numpy()[0, 0]))

                fig0.canvas.draw()
                fig1.canvas.draw()
                fig2.canvas.draw()
                fig3.canvas.draw()
                fig4.canvas.draw()

                plt.pause(0.01)
                print('left rot: ' + str(left_rotation.cpu().detach().numpy()[0]))
                print('left trans: ' + str(left_translation.cpu().detach().numpy()[0]))
                print('Pose loss ' + str(pose_loss_1_val.item() + pose_loss_2_val.item()))
                print('Step: ' + str(step) + '/' + str(math.ceil(len(dataset) // batch_size)))
                print('Total loss: ' + str(loss.item()))
                print('Epoch: ' + str(epoch))
                torch.save(model.state_dict(), 'weights')
        torch.save(model.state_dict(), f'weights_epoch_{str(epoch)}')
    cleanup()



if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    #processes = mp.spawn(run, args=(world_size, 10, 4), nprocs=world_size)
    #processes.join()

    run(0, 1, 100, 24)
