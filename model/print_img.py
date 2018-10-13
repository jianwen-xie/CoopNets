from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.misc
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import math


def main():
    output_folder = '../output_mnist_new_setting_1/mnist'

    # print images
    data = sio.loadmat(os.path.join(output_folder, 'parzon_syn_dat.mat'))
    images = data['samples_des']
    images = images[:100]
    images = np.reshape(images, (images.shape[0], 28, 28, 1))
    [num_images, image_size] = images.shape[0:2]
    col_num, row_num = 10, 10
    margin_syn = 2
    num_cells = int(math.ceil(num_images / (col_num * row_num)))
    cell_image = np.zeros((num_cells, row_num * image_size + (row_num - 1) * margin_syn,
                           col_num * image_size + (col_num - 1) * margin_syn, images.shape[-1]))
    for i in range(num_images):
        cell_id = int(math.floor(i / (col_num * row_num)))
        idx = i % (col_num * row_num)
        ir = int(math.floor(idx / col_num))
        ic = idx % col_num
        temp = images[i]
        if len(temp.shape) == 2:
            temp = np.expand_dims(temp, axis=2)
        cell_image[cell_id, (image_size + margin_syn) * ir:image_size + (image_size + margin_syn) * ir,
        (image_size + margin_syn) * ic:image_size + (image_size + margin_syn) * ic, :] = temp

    output_file = os.path.join(output_folder, 'syn.png')
    scipy.misc.imsave(output_file, np.squeeze(cell_image))

    # print curves
    data = sio.loadmat(os.path.join(output_folder, 'parzon.mat'))
    parzon_des, parzon_gen = data['parzon_des_mean'], data['parzon_gen_mean']
    output_file = os.path.join(output_folder, 'curve.png')
    fontsize = 16

    plt.figure()
    ax = plt.subplot(111)
    plt.plot(np.asarray(range(315)), np.squeeze(parzon_des)[:315], label='Descriptor')
    plt.plot(np.asarray(range(315)), np.squeeze(parzon_gen)[:315], label='Generator')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel('Parzon window-based log-likelihood', fontsize=fontsize)
    # plt.xlabel('Epoch')
    # plt.ylabel('Parzon window-based log-likelihood')
    plt.legend()
    plt.savefig(output_file)


if __name__ == "__main__":
    main()