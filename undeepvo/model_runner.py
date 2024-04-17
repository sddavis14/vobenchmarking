#!/usr/bin/env python3

# suppress deprecation warnings in kornia
import warnings;

warnings.warn = lambda *args, **kwargs: None

import torch
from model.full_model import UnDeepVO
from matplotlib import pyplot as plt

import numpy as np
from losses.helpers import generate_transformation
import time
from PIL import Image


def run():
    device = torch.device('cpu')
    model = UnDeepVO().to(device)

    state_dict = torch.load('weights_epoch_0', map_location=device)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    img_path = '/Users/spencer/vobenchmarking/undeepvo/four_seasons/training/recording_2020-03-24_17-36-22/undistorted_images/cam0/1585067971264306176.png'
    next_img_path = '/Users/spencer/vobenchmarking/undeepvo/four_seasons/training/recording_2020-03-24_17-36-22/undistorted_images/cam0/1585067971396306432.png'

    img = Image.open(img_path).resize((512, 256))
    next_img = Image.open(next_img_path).resize((512, 256))

    cropped_height = 256 - 64

    img_numpy = (np.asarray(img, dtype=np.float32)[0:cropped_height] / 127.5) - 1
    next_img_numpy = (np.asarray(next_img, dtype=np.float32)[0:cropped_height] / 127.5) - 1

    img_numpy = np.expand_dims(np.repeat(np.expand_dims(img_numpy, axis=0), 3, axis=0), axis=0)
    next_img_numpy = np.expand_dims(np.repeat(np.expand_dims(next_img_numpy, axis=0), 3, axis=0), axis=0)

    image_input = torch.from_numpy(img_numpy).to(device)
    image_input_2 = torch.from_numpy(next_img_numpy).to(device)

    output = model(image_input, image_input_2)

    t1 = time.time()
    output = model(image_input, image_input_2)
    t2 = time.time()

    print(t2 - t1)

    with torch.no_grad():
        fig2 = plt.figure()
        plt.title('depth')
        right = plt.imshow(np.log10(output[0].cpu().numpy()[0, 0]), interpolation='nearest', cmap='inferno', vmin=0, vmax=2)
        print('pose: ' + str(output[1][0].cpu().numpy()[0]))
        print('pose2: ' + str(output[1][1].cpu().numpy()[0]))

        fig2.canvas.draw()

    plt.show()

    #onnx_model = torch.onnx.dynamo_export(model, image_input, image_input_2)
    #onnx_model.save("undeepvo.onnx")


if __name__ == "__main__":
    run()
