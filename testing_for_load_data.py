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
from keras import backend as K
K.set_image_data_format('channels_last')
from skimage import exposure

from os.path import join, basename
from os import makedirs
from custom_data_aug import augmentImages

from numpy.random import rand, shuffle
import SimpleITK as sitk

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

debug = 0

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
    return np.stack(eq_img, axis=2)

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

    for folder in get_sub_folder(data_dir):
        # This part get the image inside the patient data which consist of some folder for MRI image modalities
        print(folder)
        data[folder] = {}
        for sub_folder in get_sub_folder(os.path.join(data_dir, folder)):
            image_type = get_image_type_from_folder_name(sub_folder)

            if image_type == '.MR_4DPWI':
                continue
            path = os.path.join(data_dir, folder, sub_folder)
            filename = next(filename for filename in os.listdir(path) if get_extension(filename) == '.nii')
            path = os.path.join(path, filename)

            im = nib.load(path, mmap = False)
            image = im.get_fdata()

            # print(image.shape[0])
            # height, _, slice_num = image.shape
            # If the pixel size is 192 do this
            index = np.arange(0, image.shape[2], 1)
            if image.shape[0] == 192:
                if image_type == '.MR_ADC':
                    print('currently getting the image of ADC')

                    for j in index:
                        image_list.append(np.array(image[:, :, j:j+1:2].astype(np.float32)))

                    print('ADC has been gathered')

                if image_type == '.OT':
                    print('currently getting the image of Ground Truth')
                    for j in index:
                        ground_truth_list.append(np.array(image[:, :, j:j+1:2].astype(np.float32)))

                    print('the image of Ground Truth has been sliced')

                else:
                    continue

            # If needed, for other pixel size do resize in this section
            # For now just continue
            else:
                continue
    # Do split here
    image_all = np.concatenate(image_list, axis=2)
    ground_truth_all = np.concatenate(ground_truth_list, axis=2)

    return image_all, ground_truth_all

@threadsafe_generator
def generate_train_batches(image, ground_truth, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1, stride=1, shuff=1, aug_data=1):
    # create placeholders for training data
    # (image_shape[1], img_shape[2], args.slices)
    # the shape of input is [192,192,args.slices]

    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.uint8)

    while True:

        count = 0

        if numSlices == 1:
            subSampAmt = 0
        elif subSampAmt == -1 and numSlices > 1:
            np.random.seed(None)
            subSampAmt = int(rand(1) * (369 * 0.05))

        for j in range(0, 99):
            if not np.any(ground_truth[:, :, 66:67:1]):
                continue

            # insert the img_batch from image_data for train where took all the [:, :, x:y:z]
            # x starting index, y stop index, z how much step it takes
            # where x:y:z implied that the slice say, j=0, numslices=1, subsampamt=0, take all [:,:, 0:1:1]
            # for next j=1, take all [:,:, 1:2:1]
            # for j=10, take all [:,:, 10:11:1]
            if img_batch.ndim == 4:
                img_batch[count, :, :, :] = image[:, :, 66:67:1]
                mask_batch[count, :, :, :] = ground_truth[:, :, 66:67:1]
            elif img_batch.ndim == 5:
                # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                img_batch[count, :, :, :, 0] = image[:, :, 66:67:1]
                mask_batch[count, :, :, :, 0] = ground_truth[:, :, 66:67:1]
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
                    plt.savefig(join('D:\Engineering Physics\Skripsi\Program\Ischemic Stroke Segmentation', 'logs', 'ex_train.png'), format='png', bbox_inches='tight')
                    plt.close()
                if net.find('caps') != -1:  # if the network is capsule/segcaps structure
                    # This part multiply each element between them
                    # Maybe to produce segmented image
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
                         stride=1, downSampAmt=1, shuff=1):
    # Create placeholders for validation
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.uint8)

    while True:

        count = 0
        if numSlices == 1:
            subSampAmt = 0
        elif subSampAmt == -1 and numSlices > 1:
            np.random.seed(None)
            subSampAmt = int(rand(1) * 527 * 0.05)

        for j in range(0, 19):
            if not np.any(gt_val_list[:, :, 66:67:1]):
                continue
            if img_batch.ndim == 4:
                img_batch[count, :, :, :] = val_list[:, :, 66:67:1]
                mask_batch[count, :, :, :] = gt_val_list[:, :, 66:67:1]
            elif img_batch.ndim == 5:
                # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                img_batch[count, :, :, :, 0] = val_list[:, :, 66:67:1]
                mask_batch[count, :, :, :, 0] = gt_val_list[:, :, 66:67:1]
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

# FOR TEST

@threadsafe_generator
def generate_test_batches(images_test, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                          stride=1, downSampAmt=1):
    # Create placeholders for testing
    logging.info('\nload_3D_data.generate_test_batches')
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    count = 0
    if numSlices == 1:
        subSampAmt = 0
    elif subSampAmt == -1 and numSlices > 1:
        np.random.seed(None)
        subSampAmt = int(rand(1) * (images_test.shape[2] * 0.05))

    for j in range(0,99):
        if img_batch.ndim == 4:
            img_batch[count, :, :, :] = images_test[:, :, 26:27:1]
        elif img_batch.ndim == 5:
            # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
            img_batch[count, :, :, :, 0] = images_test[:, :, 26:27:1]
        else:
            logging.error('Error this function currently only supports 2D and 3D data.')
            exit(0)

        count += 1
        if count % batchSize == 0:
            count = 0
            yield (img_batch)

    if count != 0:
        yield (img_batch[:count, :, :, :])
