# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# THIS FILE HAS BEEN MODIFIED FOR THE NWAP WORKSHOP

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

# TODO:
# Allow for flipping image distortion (that is, modify the label accordingly). Similarly, fancier augmentations of the
# data also require altering the labels


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 15212 #15212 total images available
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1500
DATA_DIRS = ["TimeStampedOriginal/centerImages/"]  # NOTE: Change this accordingly to match whether the data is for
# training or testing. Should probably modify to use python flags
LOG_FILES = ["TimeStampedOriginal/approximatedStamps.txt"]
GLOBALHEIGHT = 480
GLOBALWIDTH = 640


def get_names_and_labels(data_dirs):
    allfiles = []
    alllabels = []
    i = 0
    while i<len(data_dirs):
        data_dir = data_dirs[i]
        log_file = LOG_FILES[i]
        labelfile = open(log_file, "r")
        line = labelfile.readline()
        files = []
        labels = []
        while len(line) > 0:
            totalstring = line.split(" ")
            files.append(data_dir + totalstring[0])
            labels.append(float(totalstring[1]))
            line = labelfile.readline()
        allfiles = allfiles + files
        alllabels = alllabels + labels
        i+=1
    return allfiles, alllabels

def read_image(filename_queue):
    class ImageRecord(object):
        pass

    result = ImageRecord()
    # tensorkey,image_file=image_reader.read(filename_queue)

    image_file = tf.read_file(filename_queue[0])
    result.label = filename_queue[1]
    result.uint8image = tf.image.decode_png(image_file)
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: 4D tensor of [batch_size, height, width, 3] size.
      labels: 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, then read 'batch_size' images + labels from the example queue.
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=16,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=16,
            capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dirs, batch_size):  # MUST SET HEIGHT AND WIDTH
    """Construct distorted input for training using the Reader ops.
    Args:
      data_dir: Path to the data directory.
      batch_size: Number of images per batch.
    Returns:
      images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: 1D tensor of [batch_size] size.
    """

    filenamelist, labellist = get_names_and_labels(data_dirs)
    images = tf.convert_to_tensor(filenamelist, dtype=tf.string)
    labels = tf.convert_to_tensor(labellist, dtype=tf.float32)

    input_queue = tf.train.slice_input_producer([images, labels])
    read_input = read_image(input_queue)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    # Image processing for training the network. Random distortions applied to the image.

    # Randomly flip the image horizontally:
    #   distorted_image = tf.image.random_flip_left_right(reshaped)
    # Cannot do this easily because we must change the label accordingly. Needs additional work

    print("Distorting images")
    sys.stdout.flush()

    # Because these operations are not commutative, consider randomizing order of their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(reshaped_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([GLOBALHEIGHT, GLOBALWIDTH, 3])
    read_input.label.set_shape([])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    sys.stdout.flush()

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

def inputs(data_dir, batch_size):
    """Construct distorted input for training using the Reader ops.
    Args:
      data_dir: Path to the data directory.
      batch_size: Number of images per batch.
    Returns:
      images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: 1D tensor of [batch_size] size.
    """

    filenamelist, labellist = get_names_and_labels(data_dir)
    images = tf.convert_to_tensor(filenamelist, dtype=tf.string)
    labels = tf.convert_to_tensor(labellist, dtype=tf.float32)

    input_queue = tf.train.slice_input_producer([images, labels])
    read_input = read_image(input_queue)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    # Subtract the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    float_image.set_shape([GLOBALHEIGHT, GLOBALWIDTH, 3])
    read_input.label.set_shape([])

    # Ensure that random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d images before starting to eval. '
          'This will take a few minutes.' % min_queue_examples)
    sys.stdout.flush()

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)
