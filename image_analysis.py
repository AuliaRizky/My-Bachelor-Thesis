import numpy as np
from numpy.random import rand, shuffle
from matplotlib import pyplot as plt
from os.path import join


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


def analyze_img_data(img_train, img_val, img_test, images):
    train_num = 0
    val_num = 0
    test_num = 0
    images_num = 0

    for i in range(0, len(img_train), 1):
        if img_train[i].shape[2] == 19:
            train_num += 1
        else:
            continue
    for i in range(0, len(img_val), 1):
        if img_val[i].shape[2] == 19:
            val_num += 1
        else:
            continue
    for i in range(0, len(img_test), 1):
        if img_test[i].shape[2] == 19:
            test_num += 1
        else:
            continue
    for i in range(0, len(images), 1):
        if images[i].shape[2] == 19:
            images_num += 1
        else:
            continue
    print("Number of Train Image Slice 19: ", train_num)
    print("Number of Val Image Slice 19  : ", val_num)
    print("Number of Test Image Slice 19 : ", test_num)
    print("Number of All Images          : ", images_num)


def roi_histo_analysis(images, g_t):
    img_id = np.arange(0, len(images), 1)
    shuffle(img_id)
    img_slice = 0
    histo_img = []
    histo_gt = []
    i = 0
    while i < 5:
        while not np.any(g_t[img_id[i]][:, :, img_slice]):
            img_slice += 1
        img = images[img_id[i]][:, :, img_slice:img_slice+1:1]
        gt = g_t[img_id[i]][:, :, img_slice:img_slice+1:1]
        histo_img.append(img)
        histo_gt.append(gt)
        i += 1
    print(histo_gt[0].shape)
    for j in range(0, histo_img[0].shape[2], 1):
        image = histo_img[j] * histo_gt[j] *256
        plt.hist(image.ravel)
        plt.show()

def show_image(images, gt):
    a = 0
    while a < 3:
        plt.imshow(images[a][:, :, 10], cmap='gray')
        plt.show()
        plt.imshow(gt[a][:, :, 10], cmap='gray')
        plt.show()
        a += 1

'''def get_image_example(images, ground_truth):
    

if img_batch.ndim == 4:
    plt.imshow(np.squeeze(img_batch[0, :, :, 0]), cmap='gray')
    plt.imshow(np.squeeze(mask_batch[0, :, :, 0]), alpha=0.15)
elif img_batch.ndim == 5:
    plt.imshow(np.squeeze(img_batch[0, :, :, 0, 0]), cmap='gray')
    plt.imshow(np.squeeze(mask_batch[0, :, :, 0, 0]), alpha=0.15)
plt.savefig(
    join('D:\Engineering Physics\Skripsi\Program\Ischemic Stroke Segmentation', 'logs',
         'ex_train.png'), format='png', bbox_inches='tight')
plt.close()'''