'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.
This file is used for loading training, validation, and testing data into the models.
It is specifically designed to handle 3D single-channel medical data.
Modifications will be needed to train/test on normal 3-channel images.
=====
This program includes all functions of 3D image processing for UNet, tiramisu, Capsule Nets (capsbasic) or SegCaps(segcapsr1 or segcapsr3).
@author: Cheng-Lin Li a.k.a. Clark
@copyright:  2018 Cheng-Lin Li@Insight AI. All rights reserved.
@license:    Licensed under the Apache License v2.0. http://www.apache.org/licenses/
@contact:    clark.cl.li@gmail.com
Tasks:
    The program based on parameters from main.py to load 3D image files from folders.
    The program will convert all image files into numpy format then store training/testing images into
    ./data/np_files and training (and testing) file lists under ./data/split_list folders.
    You need to remove these two folders every time if you want to replace your training image and mask files.
    The program will only read data from np_files folders.
Data:
    MS COCO 2017 or LUNA 2016 were tested on this package.
    You can leverage your own data set but the mask images should follow the format of MS COCO or with background color = 0 on each channel.
Enhancement:
    1. Porting to Python version 3.6
    2. Remove program code cleaning
'''

from __future__ import print_function

import os
import nibabel as nib
import numpy as np
import logging
import operator
from keras import backend as K
K.set_image_data_format('channels_last')
from skimage import exposure
from sklearn.model_selection import train_test_split
from os.path import join
from custom_data_aug import augmentImages
from numpy.random import rand, shuffle
import matplotlib.pyplot as plt

from threadsafe import threadsafe_generator

# Now preprocess the data so that the shape has the same size
# The data contains:
# 17 image of (192, 192, 19)
# 6           (192, 192, 30)
# 1           (192, 192, 24)
# 10          (128, 128, 25)
# 9           (256, 256, 24)
# Consult if we need to do more processing because the total train dataset after slicing is 527 2d image
# If we resize the other data the total dataset is 993 2d image
# Do contrast normalization

debug = 1

def contrast_normalization(image, min_divisor=1e-3):
    mean = image.mean()
    std = image.std()
    if std < min_divisor:
        std = min_divisor
    return (image - mean) / std

def equalize_image(image_train):
    eq_img = []
    index = np.arange(0, image_train.shape[2], 1)
    for i in index:
        eq_img.append(exposure.equalize_hist(image_train[:, :, i]))
    return np.stack(eq_img, axis=-1)

def range_normalization(image):
    image[image > 2048] = 2048
    image /= 2048
    return image

# Credit to https://stackoverflow.com/users/3931936/losses-don
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def reduce_slice(image, up_slice, down_slice):
    image = image[:, :, up_slice - 1:image.shape[2] - down_slice:1]
    return image

def get_extension(filename):
    filename, extension = os.path.splitext(filename)
    return extension

def get_image_type_from_folder_name(sub_folder):
    image_types = ['.MR_4DPWI', '.MR_ADC', '.MR_MTT', '.MR_rCBF', '.MR_rCBV', '.MR_Tmax', '.MR_TTP', '.OT']
    return next(image_type for image_type in image_types if image_type in sub_folder)

def get_sub_folder(data_dir):
    return [sub_folder for sub_folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, sub_folder))]

def read_and_process_data(data_dir):
    # This method get all the image available for every patient data
    data = {}
    image_list = []
    ground_truth_list = []
    print('Reading Image....')
    for folder in get_sub_folder(data_dir):
        # This part get the image inside the patient data which consist of some folder for MRI image modalities
        data[folder] = {}
        for sub_folder in get_sub_folder(os.path.join(data_dir, folder)):
            image_type = get_image_type_from_folder_name(sub_folder)
            if image_type == '.MR_4DPWI':
                continue
            path = os.path.join(data_dir, folder, sub_folder)
            filename = next(filename for filename in os.listdir(path) if get_extension(filename) == '.nii')
            path = os.path.join(path, filename)

            im = nib.load(path, mmap=True)
            image = im.get_fdata()

            folder_dict = {"training_28": [6, 3], "training_30": [4, 5], "training_31": [6, 3], "training_32": [4, 5],
                           "training_33": [5, 4], "training_35": [6, 3], "training_8": [2, 1]}

            # If the pixel size is 192 do this
            if image.shape[0] == 192:
                if image.shape[2] == 30 or image.shape[2] == 24:
                    if image_type == '.MR_ADC':
                        image_slice = []
                        up_slice = folder_dict.get(str(folder))[0]
                        down_slice = folder_dict.get(str(folder))[1]

                        image = reduce_slice(range_normalization((np.array(image))),
                                             up_slice + 1,
                                             down_slice + 1)

                        for slice_num in range(0, image.shape[2]-1, 1):
                            image_slice.append(np.fliplr(cropND(image[:, :, slice_num:slice_num+1:1],
                                                                (160, 160))))
                        image_list.append(np.concatenate(image_slice, axis=-1))

                    if image_type == '.OT':

                        image_slice = []
                        up_slice = folder_dict.get(str(folder))[0]
                        down_slice = folder_dict.get(str(folder))[1]

                        image = reduce_slice(np.array(image), up_slice + 1, down_slice + 1)
                        for slice_num in range(0, image.shape[2] - 1, 1):
                            image_slice.append(np.fliplr(cropND(image[:, :, slice_num:slice_num + 1:1],
                                                      (160, 160))))
                        ground_truth_list.append(np.concatenate(image_slice, axis=-1))

                else:
                    if image_type == '.MR_ADC':
                        image_slice = []
                        for slice_num in range(0, image.shape[2], 1):
                            image_slice.append(cropND(range_normalization(image[:, :, slice_num:slice_num + 1:1]),
                                                      (160, 160)))
                        image_list.append(np.concatenate(image_slice, axis=-1))


                    if image_type == '.OT':
                        image_slice = []
                        for slice_num in range(0, image.shape[2], 1):
                            image_slice.append(cropND(image[:, :, slice_num:slice_num + 1:1],
                                                      (160, 160)))
                        ground_truth_list.append(np.concatenate(image_slice, axis=-1))

            else:
                continue

            # If needed, for other pixel size do resize in this section
            # For now just continue
    print('Image Acquisition Completed...')
    return image_list, ground_truth_list

def analyze_data(g_t_train, g_t_val, g_t_test, ground_truth):

    count_train = 0
    count_val = 0
    count_test = 0
    count = 0

    for i in range(0, len(g_t_train), 1):
        index = np.arange(0, g_t_train[0].shape[2])
        for j in index:
            if np.any(g_t_train[i][:, :, j:j + 1:1]):
                count_train += 1

    for i in range(0, len(g_t_val), 1):
        index = np.arange(0, g_t_val[0].shape[2])
        for j in index:
            if np.any(g_t_val[i][:, :, j:j + 1:1]):
                count_val += 1

    for i in range(0, len(g_t_test), 1):
        index = np.arange(0, g_t_test[0].shape[2])
        for j in index:
            if np.any(g_t_test[i][:, :, j:j + 1:1]):
                count_test += 1

    for i in range(0, len(ground_truth), 1):
        index = np.arange(0, ground_truth[0].shape[2])
        for j in index:
            if np.any(ground_truth[i][:, :, j:j + 1:1]):
                count += 1

    print('Number of slice that have ROI:')
    print('Train set               = ', count_train)
    print('Val set                 = ', count_val)
    print('Test set                = ', count_test)
    print('Total Number            = ', count_test + count_train + count_val)
    print('Should the Total Number = ', count)

def generate_train_test(images, gt_images, random_num):
    index_image_list = list(range(0, len(images), 1))
    index_gt_list = list(range(0, len(images), 1))

    index_train, index_test, _, _ = train_test_split(index_image_list, index_gt_list,
                                                                  test_size=0.125,
                                                                  random_state=random_num)
    images_train = []
    gt_train = []
    images_test = []
    gt_test = []

    for i in index_train:
        images_train.append(images[i])
        gt_train.append(gt_images[i])
    for j in index_test:
        images_test.append(images[j])
        gt_test.append(gt_images[j])

    return images_train, images_test, gt_train, gt_test

def generate_train_val(images, gt_images, random_num):
    index_image_list = list(range(0, len(images), 1))
    index_gt_list = list(range(0, len(images), 1))

    index_train, index_val, _, _ = train_test_split(index_image_list, index_gt_list,
                                                                  test_size = 0.15,
                                                                  random_state = random_num)
    images_train = []
    gt_train = []
    images_val = []
    gt_val = []

    for i in index_train:
        images_train.append(images[i])
        gt_train.append(gt_images[i])
    for j in index_val:
        images_val.append(images[j])
        gt_val.append(gt_images[j])

    return images_train, images_val, gt_train, gt_val

@threadsafe_generator
def generate_train_batches(images, ground_truth, net_input_shape, net, batchSize=1, numSlices=1,
                           subSampAmt=-1,
                           stride=1,
                           shuff=1,
                           aug_data=1, same=0, index_num=[9]):

    # create placeholders for training data
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.uint8)

    while True:

        count = 0
        for i in range(0, len(images), 1):
            image = images[i]
            g_t_image = ground_truth[i]
            # Slice Number that wanted to be taken from each patient
            if not same:
                indexes = np.arange(0, image.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
                if shuff:
                    shuffle(indexes)
            else:
                indexes = index_num

            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1:
                np.random.seed(None)
                subSampAmt = int(rand(1) * (image.shape[2] * 0.05))

            for j in indexes:
                if not np.any(g_t_image[:, :, j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]):
                    continue

                if img_batch.ndim == 4:
                    img_batch[count, :, :, :] = image[:, :, j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
                    mask_batch[count, :, :, :] = g_t_image[:, :, j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count, :, :, :, 0] = image[:, :, j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
                    mask_batch[count, :, :, :, 0] = g_t_image[:, :,
                                                    j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
                else:
                    logging.error('\nError this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if aug_data:
                        img_batch, mask_batch = augmentImages(img_batch, mask_batch)
                    if debug:
                        if img_batch.ndim == 4:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0]), alpha=0.15)
                        elif img_batch.ndim == 5:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0, 0]), alpha=0.15)
                        plt.savefig(
                            join('D:\Engineering Physics\Skripsi\Program\Ischemic Stroke Segmentation', 'logs',
                                 'ex_train.png'), format='png', bbox_inches='tight')
                        plt.close()
                    if net.find('caps') != -1:  # if the network is capsule/segcaps structure
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if aug_data:
                img_batch[:count, ...], mask_batch[:count, ...] = augmentImages(img_batch[:count, ...],
                                                                                mask_batch[:count, ...])
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count, ...], mask_batch[:count, ...])


@threadsafe_generator
def generate_val_batches(val_list, gt_val_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                         stride=1, downSampAmt=1, shuff=1, same=0, index_num = 9):
    # Create placeholders for validation
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.uint8)

    while True:
        count = 0
        for i in range(0, len(val_list), 1):
            val_image = val_list[i]
            gt_val_image = gt_val_list[i]
            # Slice Number that wanted to be taken from each patient
            if not same:
                indexes = np.arange(0, val_image.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
                if shuff:
                    shuffle(indexes)
            else:
                indexes = index_num

            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1:
                np.random.seed(None)
                subSampAmt = int(rand(1) * (val_image.shape[2] * 0.05))

            for j in indexes:
                if not np.any(gt_val_image[:, :, j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]):
                    continue
                if img_batch.ndim == 4:
                    img_batch[count, :, :, :] = val_image[:, :, j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
                    mask_batch[count, :, :, :] = gt_val_image[:, :, j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count, :, :, :, 0] = val_image[:, :,
                                                   j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
                    mask_batch[count, :, :, :, 0] = gt_val_image[:, :,
                                                    j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
                else:
                    logging.error('\nError this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if net.find('caps') != -1:
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count, ...], mask_batch[:count, ...])


@threadsafe_generator
def generate_test_batches(images_test, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                          stride=1, downSampAmt=1, same=0, index_num=[9, 10]):
    # Create placeholders for testing
    logging.info('\nload_3D_data.generate_test_batches')
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)

    count = 0
    if numSlices == 1:
        subSampAmt = 0
    elif subSampAmt == -1 and numSlices > 1:
        np.random.seed(None)
        subSampAmt = int(rand(1) * (images_test.shape[2] * 0.05))

    if not same:
        indexes = np.arange(0, images_test.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
    else:
        indexes = index_num

    for j in indexes:
        if img_batch.ndim == 4:
            img_batch[count, :, :, :] = images_test[:, :, j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
        elif img_batch.ndim == 5:
            # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
            img_batch[count, :, :, :, 0] = images_test[:, :, j:j + numSlices * (subSampAmt + 1):subSampAmt + 1]
        else:
            logging.error('Error this function currently only supports 2D and 3D data.')
            exit(0)

        count += 1
        if count % batchSize == 0:
            count = 0
            yield (img_batch)

    if count != 0:
        yield (img_batch[:count, :, :, :])
