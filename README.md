# My-Bachelor-Thesis
Segcaps Implementation for Stroke Lesion Segmentation

This works is my bachelor thesis using SegCaps developed by Lalonderodney and Cheng Shin Li Implementation

Progress of Project:

Note: 28 Jan 2019
Overfit test shows the dice hard metrics could not exceed 0.1127

Note: 29 Jan 2019
Overfit test shows dice hard increase to 0.20200 and could not exceeded.
This result achieved by resizing the image into 140 x 140.
Hyperparameter changes still not affect the results nor deeper layer.
This image size can not run using SegCapsR3. There is a mismatch between the concatenated layer. This is the product of striding.
It looks like the overfit test could achieve 100% metric but there is something that prevent the model to learn further.

The challenge is to figure out why the loss and the metrics could not maximum.
