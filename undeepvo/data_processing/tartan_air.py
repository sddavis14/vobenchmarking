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
from matplotlib import pyplot as plt


class TartanAirDataset(Dataset):
    def __init__(self, data_path='tartan_air', data_type='training'):
        print('Looking for dataset in \"' + data_path + '/' + data_type + '\"')
        glob_result = glob.glob(data_path + '/' + data_type + '/*/*/*/pose_left.txt')
        self.dataset_paths = [os.path.dirname(x) for x in glob_result]

        # List of all training images that have a consecutive image
        self.all_training_image_pairs = self.get_all_training_image_pairs()

        self.left_to_right_camera = np.array([ 1, 0, 0, -0.25,
                                               0, 1, 0, 0,
                                               0, 0, 1, 0,
                                               0, 0, 0, 1], dtype=np.float32).reshape(4, 4)

        self.original_height = 480
        self.original_width = 640

        self.scale = 0.8

        self.height = int(self.original_height * self.scale)
        self.width = int(self.original_width * self.scale)

        # pulled from the file... file format is ugly so just build the matrices directly
        self.intrinsics_matrix_left = np.array([self.scale * 320, 0, self.scale * 320,
                                                0, self.scale * 320, self.scale * 240,
                                                0, 0, 1], dtype=np.float32).reshape(3, 3)
        self.intrinsics_matrix_right = np.array([self.scale * 320, 0, self.scale * 320,
                                                 0, self.scale * 320, self.scale * 240,
                                                 0, 0, 1], dtype=np.float32).reshape(3, 3)

    def get_all_training_image_pairs(self):
        all_training_image_pairs = []
        for dataset_idx in range(len(self.dataset_paths)):
            image_count = len(glob.glob(self.dataset_paths[dataset_idx] + '/image_left/*.png'))
            for time_idx in range(image_count - 1):
                all_training_image_pairs.append((dataset_idx, time_idx))
        return all_training_image_pairs

    def get_image_pair(self, dataset_idx, image_idx):
        root = self.dataset_paths[dataset_idx]
        left_img_path = root + '/image_left/{:06d}_left.png'.format(image_idx)
        right_img_path = root + '/image_right/{:06d}_right.png'.format(image_idx)

        left_img = Image.open(left_img_path).resize((self.width, self.height))
        right_img = Image.open(right_img_path).resize((self.width, self.height))

        left_img_numpy = (np.asarray(left_img, dtype=np.float32) / 127.5) - 1
        right_img_numpy = (np.asarray(right_img, dtype=np.float32) / 127.5) - 1

        left_img_numpy = np.transpose(left_img_numpy, (2, 0, 1))
        right_img_numpy = np.transpose(right_img_numpy, (2, 0, 1))

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

