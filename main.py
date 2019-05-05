'''
    This is the main program  where we do all of our stuff
    To do train and test just run this main program
    Don't forget to parse the arguments needed!
    Developed from all other programmer that share their works as open source project
    This work based on the work of SegCaps algorithm developed by La Londe Rodney
    and Cheng Lin Li. The biggest credit goes to those guys.
    A bachelor thesis work by Aulia Rizky Hermawan
    Student of Engineering Physics, Universitas Gadjah Mada, Indonesia
'''

import argparse
from keras import backend as K

K.set_image_data_format('channels_last')

from os.path import join
from os import makedirs
from os import environ

from keras.utils import print_summary
from time import gmtime, strftime

from load_data import read_and_process_data, generate_train_test, generate_train_val, generate_train_batches
from image_analysis import analyze_data, analyze_img_data, roi_histo_analysis, show_image
from model_helper import create_model
import random
time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

def main(args):
    # Ensure training and testing are not all turned off
    assert (args.train or args.test), 'Cannot have train or tes as 0'

    # Load the data
    images, ground_truth = read_and_process_data(args.data_root_dir, args.size)

    images_train_val, images_test, g_t_train_val, g_t_test = generate_train_test(images, ground_truth,
                                                                         random_num=random.randint(1, 1001))
    images_train, images_val, g_t_train, g_t_val = generate_train_val(images_train_val, g_t_train_val,
                                                                      random_num=random.randint(1, 1001))

    show = 0
    if show:
        show_image(images_train, g_t_train)
        show_image(images_val, g_t_val)

    input_shape = (images[0].shape[0], images[0].shape[1], 1)

    analyze = 1
    if analyze:
        analyze_data(g_t_train, g_t_val, g_t_test, ground_truth)
    if analyze:
        analyze_img_data(images_train, images_val, images_test, images)

    if args.num_pat != 0:
        images_train_new = []
        g_t_train_new = []
        for i in range(0, args.num_pat, 1):
            images_train_new.append(images_train[i])
            g_t_train_new.append(g_t_train[i])

        images_train = images_train_new
        g_t_train = g_t_train_new


    # Create the model for training and testing
    # model_list = [0] train_model, [1] eval_model
    model_list = create_model(args=args, input_shape=input_shape)

    print_summary(model=model_list[0], positions=[.38, .65, .75, 1.])

    args.output_name = 'split-' + str(args.split_num) + '_batch-' + str(args.batch_size) + \
                       '_shuff-' + str(args.shuffle_data) + '_aug-' + str(args.aug_data) + \
                       '_loss-' + str(args.loss) + '_--slic-' + str(args.slices) + \
                       '_sub-' + str(args.subsamp) + '_strid-' + str(args.stride) + \
                       '_lr-' + str(args.initial_lr) + '_recon-' + str(args.recon_wei)
    args.time = time

    args.check_dir = join('D:\Engineering Physics\Skripsi\Program\Ischemic Stroke Segmentation', 'saved_models')
    try:
        makedirs(args.check_dir)
    except:
        pass

    args.log_dir = join('D:\Engineering Physics\Skripsi\Program\Ischemic Stroke Segmentation', 'logs')
    try:
        makedirs(args.log_dir)
    except:
        pass

    args.tf_log_dir = join('D:\Engineering Physics\Skripsi\Program\Ischemic Stroke Segmentation', 'tf_logs')
    try:
        makedirs(args.tf_log_dir)
    except:
        pass

    args.output_dir = join('D:\Engineering Physics\Skripsi\Program\Ischemic Stroke Segmentation', 'plots', args.net)
    try:
        makedirs(args.output_dir)
    except:
        pass

    if args.train:
        from train import train
        # Run training
        train(args, images_train, images_val, g_t_train, g_t_val, model_list[0], input_shape)

    if args.test:
        from test import test
        # Run testing
        test(args, [images[6]], [ground_truth[6]], model_list, input_shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on Medical Data')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory for your data.')
    parser.add_argument('--weights_path', type=str, default='',
                        help='/path/to/trained_model.hdf5 from root. Set to "" for none.')
    parser.add_argument('--split_num', type=int, default=0,
                        help='Which training split to train/test on.')
    parser.add_argument('--net', type=str.lower, default='segcapsbasic',
                        choices=['segcapsr3', 'segcapsr1', 'segcapsbasic', 'devsegcaps','unet'],
                        help='Choose your network.')
    parser.add_argument('--train', type=int, default=1, choices=[0, 1],
                        help='Set to 1 to enable training.')
    parser.add_argument('--test', type=int, default=1, choices=[0, 1],
                        help='Set to 1 to enable testing.')
    parser.add_argument('--shuffle_data', type=int, default=1, choices=[0, 1],
                        help='Whether or not to shuffle the training data (both per epoch and in slice order.')
    parser.add_argument('--aug_data', type=int, default=0, choices=[0, 1],
                        help='Whether or not to use data augmentation during training.')
    parser.add_argument('--loss', type=str.lower, default='dice',
                        choices=['bce', 'bce_dice', 'w_bce', 'dice', 'mar', 'w_mar'],
                        help='Which loss to use. "bce" and "w_bce": unweighted and weighted binary cross entropy'
                             '"dice": soft dice coefficient, "mar" and "w_mar": unweighted and weighted margin loss.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training/testing.')
    parser.add_argument('--initial_lr', type=float, default=0.01,
                        help='Initial learning rate for Adam.')
    parser.add_argument('--recon_wei', type=float, default=131.072,
                        help="If using capsnet: The coefficient (weighting) for the loss of decoder")
    parser.add_argument('--size', type=int, default=160,
                        help='The size of the image.')
    parser.add_argument('--slices', type=int, default=1,
                        help='Number of slices to include for training/testing.')
    parser.add_argument('--subsamp', type=int, default=-1,
                        help='Number of slices to skip when forming 3D samples for training. Enter -1 for random '
                             'subsampling up to 5% of total slices.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Number of slices to move when generating the next sample.')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training. 0: Silent, 1: per iteration, 2: per epoch.')
    parser.add_argument('--save_raw', type=int, default=1, choices=[0, 1],
                        help='Enter 0 to not save, 1 to save.')
    parser.add_argument('--save_seg', type=int, default=1, choices=[0, 1],
                        help='Enter 0 to not save, 1 to save.')
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix to append to saved CSV.')
    parser.add_argument('--thresh_level', type=float, default=0.,
                        help='Enter 0.0 for otsu thresholding, else set value')
    parser.add_argument('--compute_dice', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_jaccard', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_assd', type=int, default=0,
                        help='0 or 1')
    parser.add_argument('--which_gpus', type=str, default="0",
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --which_gpus arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')
    parser.add_argument('--num_pat', type=int, default=0,
                        help='0 means all data set. If else than all data set number of patient,' 
                             'insert number of patient.')
    parser.add_argument('--same', type=int, default=0, choices=[0, 1],
                        help='Same indicate that data set feeded contain same slice for each patient. 0 no 1 yes')
    parser.add_argument('--index_num', nargs='+', type=int, default=[0],
                        help='insert the slice that we want to use.')

    arguments = parser.parse_args()

    if arguments.which_gpus == -2:
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = ""
    elif arguments.which_gpus == '-1':
        assert (
                    arguments.gpus != -1), 'Use all GPUs option selected under --which_gpus, with this option the user MUST ' \
                                           'specify the number of GPUs available with the --gpus option.'
    else:
        arguments.gpus = len(arguments.which_gpus.split(','))
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = str(arguments.which_gpus)

    if arguments.gpus > 1:
        assert arguments.batch_size >= arguments.gpus, 'Error: Must have at least as many items per batch as GPUs ' \
                                                       'for multi-GPU training. For model parallelism instead of ' \
                                                       'data parallelism, modifications must be made to the code.'

    main(arguments)