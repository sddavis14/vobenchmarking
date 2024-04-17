import pylab as pl
import torch
import numpy as np
import PIL.Image as Image
import glob
import os.path
from matplotlib import pyplot as plt
import math
import random
from torch.utils.data import Dataset

class FourSeasonsDataset(Dataset):
    def __init__(self, data_path='four_seasons'):
        print('Looking for dataset in \"' + data_path + '\"')
        glob_result = glob.glob(data_path + '/training/recording*/times.txt')
        self.dataset_paths = [os.path.dirname(x) for x in glob_result]
        self.dataset_times = [np.loadtxt(x + '/times.txt', dtype=int) for x in self.dataset_paths]
        
        # List of all training images that have a consecutive image
        self.all_training_image_pairs = self.get_all_training_image_pairs()

        self.remaining_batch_training_pairs = self.all_training_image_pairs.copy()

        self.left_to_right_camera = np.loadtxt(data_path + '/calibration/undistorted_calib_stereo.txt',
                                               dtype=np.float32)

        self.original_height = 400
        self.original_width = 800
        self.bottom_crop = 100

        self.scale = 0.64

        self.height = int(self.original_height * self.scale)
        self.width = int(self.original_width * self.scale)

        # pulled from the file... file format is ugly so just build the matrices directly
        self.intrinsics_matrix_left = np.array([self.scale * 501.4757919305817, 0, self.scale * 421.7953735163109,
                                                0, self.scale * 501.4757919305817, self.scale * 167.65799492501083,
                                                0, 0, 1], dtype=np.float32).reshape(3, 3)
        self.intrinsics_matrix_right = np.array([self.scale * 501.4757919305817, 0, self.scale * 421.7953735163109,
                                                 0, self.scale * 501.4757919305817, self.scale * 167.65799492501083,
                                                 0, 0, 1], dtype=np.float32).reshape(3, 3)

    def get_all_training_image_pairs(self):
        all_training_image_pairs = []
        for dataset_idx in range(len(self.dataset_paths)):
            for time_idx in range(self.dataset_times[dataset_idx].shape[0] - 1):
                all_training_image_pairs.append((dataset_idx, time_idx))
        return all_training_image_pairs

    def get_image_pair(self, dataset_idx, image_idx):
        timestamp = self.dataset_times[dataset_idx][image_idx][0]
        root = self.dataset_paths[dataset_idx] + '/undistorted_images/'
        left_img_path = root + f'cam0/{timestamp}.png'
        right_img_path = root + f'cam1/{timestamp}.png'

        left_img = Image.open(left_img_path).resize((self.width, self.height))
        right_img = Image.open(right_img_path).resize((self.width, self.height))

        cropped_height = int( self.height - (self.bottom_crop * self.scale))

        left_img_numpy = (np.asarray(left_img, dtype=np.float32)[0:cropped_height] / 127.5) - 1
        right_img_numpy = (np.asarray(right_img, dtype=np.float32)[0:cropped_height] / 127.5) - 1

        left_img_numpy = np.repeat(np.expand_dims(left_img_numpy, axis=0), 3, axis=0)
        right_img_numpy = np.repeat(np.expand_dims(right_img_numpy, axis=0), 3, axis=0)

        return left_img_numpy, right_img_numpy
    
    def __len__(self):
        return len(self.all_training_image_pairs)
    
    def __getitem__(self, idx):
        dataset_idx, img_idx = self.all_training_image_pairs[idx]

        left, right = self.get_image_pair(dataset_idx, img_idx)
        left_next, right_next = self.get_image_pair(dataset_idx, img_idx + 1)

        left_tensor = torch.from_numpy(np.squeeze(left))
        right_tensor = torch.from_numpy(np.squeeze(right))
        left_next_tensor = torch.from_numpy(np.squeeze(left_next))
        right_next_tensor = torch.from_numpy(np.squeeze(right_next))

        return left_tensor, right_tensor, left_next_tensor, right_next_tensor
    
    def baseline(self):
        return math.fabs(self.left_to_right_camera[0, 3])

    def focal_length(self):
        return self.intrinsics_matrix_left[0, 0]

    def left_intrinsics(self):
        return self.intrinsics_matrix_left

    def right_intrinsics(self):
        return self.intrinsics_matrix_right

    def left_to_right_camera_extrinsics(self):
        return self.left_to_right_camera

    def validation_batch_generator(self, batch_size):
        yield 0
