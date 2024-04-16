import pylab as pl
import torch
import numpy as np
import PIL.Image as Image
import glob
import os.path
from matplotlib import pyplot as plt
import math
import random


class FourSeasonsDataset():
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

        self.scale = 0.32

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

        left_img_numpy = (np.asarray(left_img, dtype=np.float32) / 127.5) - 1
        right_img_numpy = (np.asarray(right_img, dtype=np.float32) / 127.5) - 1

        left_img_numpy = np.repeat(np.expand_dims(left_img_numpy, axis=0), 3, axis=0)
        right_img_numpy = np.repeat(np.expand_dims(right_img_numpy, axis=0), 3, axis=0)

        return left_img_numpy, right_img_numpy

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

    def training_batch_generator(self, batch_size):
        left_image_batch = np.empty((batch_size, 3,  self.height, self.width), dtype=np.float32)
        right_image_batch = np.empty((batch_size, 3,  self.height, self.width), dtype=np.float32)
        right_next_image_batch = np.empty((batch_size, 3, self.height, self.width), dtype=np.float32)
        left_next_image_batch = np.empty((batch_size, 3, self.height, self.width), dtype=np.float32)

        while True:
            if len(self.remaining_batch_training_pairs) == 0:
                self.remaining_batch_training_pairs = self.all_training_image_pairs.copy()
                print('Finished an epoch.')

            print('Remaining training pairs ' + str(len(self.remaining_batch_training_pairs)))

            for idx in range(batch_size):
                random_image = random.choice(self.remaining_batch_training_pairs)
                self.remaining_batch_training_pairs.remove(random_image)

                # Grab the next consecutive image after the randomly chosen one
                # This is always ok since the set never includes the last image
                next_random_image = (random_image[0], random_image[1] + 1)

                left, right = self.get_image_pair(random_image[0], random_image[1])
                left_next, right_next = self.get_image_pair(next_random_image[0], next_random_image[1])

                left_image_batch[idx] = left
                right_image_batch[idx] = right
                left_next_image_batch[idx] = left_next
                right_next_image_batch[idx] = right_next

            yield left_image_batch, right_image_batch, left_next_image_batch, right_next_image_batch,

    def validation_batch_generator(self, batch_size):
        yield 0
