

'''
    This is the Pre processing Pipeline before the data set feed into the machine learning.
    This Part consist of:
    - Intensity Normalization, due to the arbitrary nature of MRI imaging. Every image differ in term of intensity range.
      To do intensity normalization, we use piece wise histogram matching method proposed by Nyul et al. (2000).
    - Histogram Equalization, to increase the appearance of stroke lesion. This method shows the increased the region of
      lesion so that the algorithm could detect easier.
'''

from __future__ import print_function

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from medpy.filter import IntensityRangeStandardization

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
            if image.shape[0] == 192:

                if image_type == '.MR_ADC':
                    print('currently getting the image of ADC')
                    image_list.append(np.array(image[:, :, :].astype(np.float32)))

                    print('ADC has been gathered')

                if image_type == '.OT':
                    print('currently getting the image of Ground Truth')
                    ground_truth_list.append(np.array(image[:, :, :].astype(np.float32)))

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

    return image_list, ground_truth_list, image_all, ground_truth_all

data_dir = "D:/Engineering Physics/Skripsi/Program/Ischemic Stroke Segmentation/train_data"
images, gt, images_all, gt_all = read_and_process_data(data_dir)
print(images[0].shape)
print(len(images))
plt.imshow(images_all[:, :, 11], cmap='gray')
plt.show()

plt.imshow(gt_all[:, :, 11], cmap='gray')
plt.show()

# Do intensity normalization

irs = IntensityRangeStandardization()

for i in range(0,len(images)):
    trained_model = irs.train_transform(images[i], surpress_mapping_check=True)





