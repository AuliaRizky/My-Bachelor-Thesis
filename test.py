'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.
This file is used for testing models. Please see the README for details about testing.
'''

from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join
from os import makedirs
import csv
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters
from metrics import dc, jc, assd

from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import print_summary

from load_data import generate_test_batches


def threshold_mask(raw_output, threshold):
    if threshold == 0:
        try:
            threshold = filters.threshold_otsu(raw_output)
        except:
            threshold = 0.5

    print('\tThreshold: {}'.format(threshold))

    raw_output[raw_output > threshold] = 1
    raw_output[raw_output < 1] = 0

    all_labels = measure.label(raw_output)
    props = measure.regionprops(all_labels)
    props.sort(key=lambda x: x.area, reverse=True)
    thresholded_mask = np.zeros(raw_output.shape)

    if len(props) >= 2:
        if props[0].area / props[1].area > 5:  # if the largest is way larger than the second largest
            thresholded_mask[all_labels == props[0].label] = 1  # only turn on the largest component
        else:
            thresholded_mask[all_labels == props[0].label] = 1  # turn on two largest components
            thresholded_mask[all_labels == props[1].label] = 1
    elif len(props):
        thresholded_mask[all_labels == props[0].label] = 1

    thresholded_mask = scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)

    return thresholded_mask


def test(args, images_test, gt_test, model_list, net_input_shape):
    if args.weights_path == '':
        weights_path = join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5')
    else:
        weights_path = join(args.weights_path)

    output_dir = join('D:\Engineering Physics\Skripsi\Program\Ischemic Stroke Segmentation', 'results', args.net)
    raw_out_dir = join(output_dir, 'raw_output')
    fin_out_dir = join(output_dir, 'final_output')
    fig_out_dir = join(output_dir, 'qual_figs')
    try:
        makedirs(raw_out_dir)
    except:
        pass
    try:
        makedirs(fin_out_dir)
    except:
        pass
    try:
        makedirs(fig_out_dir)
    except:
        pass

    if len(model_list) > 1:
        eval_model = model_list[1]
    else:
        eval_model = model_list[0]
    try:
        eval_model.load_weights(weights_path)
    except:
        print('Unable to find weights path. Testing with random weights.')
    print_summary(model=eval_model, positions=[.38, .65, .75, 1.])

    # Set up placeholders
    outfile = ''
    if args.compute_dice:
        dice_arr = np.zeros(len(images_test))
        outfile += 'dice_'
    if args.compute_jaccard:
        jacc_arr = np.zeros(len(images_test))
        outfile += 'jacc_'
    if args.compute_assd:
        assd_arr = np.zeros(len(images_test))
        outfile += 'assd_'

    # Testing the network
    print('Testing... This will take some time...')

    with open(join(output_dir, args.save_prefix + outfile + 'scores.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        row = ['Scan Name']
        if args.compute_dice:
            row.append('Dice Coefficient')

        if args.compute_jaccard:
            row.append('Jaccard Index')

        if args.compute_assd:
            row.append('Average Symmetric Surface Distance')

        writer.writerow(row)

        for i in range(0, len(images_test) - 1, 1):
            gt_test_image = gt_test[i]
            test_image = images_test[i]
            output_array = eval_model.predict_generator(generate_test_batches(test_image,
                                                                              net_input_shape,
                                                                              batchSize=1,
                                                                              numSlices=args.slices,
                                                                              subSampAmt=0,
                                                                              stride=1, same=args.same,
                                                                              index_num=args.index_num),
                                                        steps=1, max_queue_size=10, workers=4,
                                                        use_multiprocessing=False, verbose=1)

            if args.net.find('caps') != -1:
                output = output_array[0][:, :, :, 0]
                #recon = output_array[1][:, :, :, 0]
            else:
                output = output_array[:, :, :, 0]

            print(output.shape)
            output = np.rollaxis(output, 0, 3)
            print(output.shape)

            plt.imshow(output[:, :, 0], cmap='gray')
            plt.show()

            plt.imshow(output[:, :, 0], cmap='gray')
            plt.show()

            # Plot Qualitative Figure
            print('Creating Qualitative Figure for Quick Reference')
            f, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(test_image[:, :, test_image.shape[2] // 3], alpha=1, cmap='gray')
            ax[0].imshow(output[:, :, 0], alpha=0.5, cmap='Blues')
            ax[0].imshow(gt_test_image[:, :, test_image.shape[2] // 3], alpha=0.2, cmap='Reds')
            ax[0].set_title('Slice {}/{}'.format(test_image.shape[2] // 3, test_image.shape[2]))
            ax[0].axis('off')

            ax[1].imshow(test_image[:, :, test_image.shape[2] // 2], alpha=1, cmap='gray')
            ax[1].imshow(output[:, :, 0], alpha=0.5, cmap='Blues')
            ax[1].imshow(gt_test_image[:, :, test_image.shape[2] // 2], alpha=0.2, cmap='Reds')
            ax[1].set_title('Slice {}/{}'.format(test_image.shape[2] // 2, test_image.shape[2]))
            ax[1].axis('off')

            ax[2].imshow(test_image[:, :, test_image.shape[2] // 2 + test_image.shape[2] // 4], alpha=1, cmap='gray')
            ax[2].imshow(output[:, :, 0], alpha=0.5, cmap='Blues')
            ax[2].imshow(gt_test_image[:, :, test_image.shape[2] // 2 + test_image.shape[2] // 4], alpha=0.2, cmap='Reds')
            ax[2].set_title('Slice {}/{}'.format(test_image.shape[2] // 2 + test_image.shape[2] // 4, test_image.shape[2]))
            ax[2].axis('off')

            fig = plt.gcf()
            fig.suptitle('Images of Qualitative Segmentation')

            plt.savefig(join(fig_out_dir, args.time + '_qual_fig' + '.png'),
                        format='png', bbox_inches='tight')
            plt.close('all')

            row = ['Average Score']
            if args.compute_dice:
                print('Computing Dice')
                dice_arr[i] = dc(output, gt_test_image)
                print('\tDice: {}'.format(dice_arr[i]))
                row.append(dice_arr[i])
            if args.compute_jaccard:
                print('Computing Jaccard')
                jacc_arr[i] = jc(output, gt_test_image)
                print('\tJaccard: {}'.format(jacc_arr[i]))
                row.append(jacc_arr[i])
            if args.compute_assd:
                print('Computing ASSD')
                assd_arr[i] = assd(output, gt_test_image, connectivity=1)
                print('\tASSD: {}'.format(assd_arr[i]))
                row.append(assd_arr[i])

            writer.writerow(row)

        row = ['Average Scores']
        if args.compute_dice:
            row.append(np.mean(dice_arr))
        if args.compute_jaccard:
            row.append(np.mean(jacc_arr))
        if args.compute_assd:
            row.append(np.mean(assd_arr))
        writer.writerow(row)

print('Done.')