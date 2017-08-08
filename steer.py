# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import steer_input

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Basic model parameters.
BATCH_SIZE = 128
DATA_DIRS = steer_input.DATA_DIRS
EVAL_DATA_DIR = "data/eurotruck/1/data/" #"TimeStampedOriginal/centerImages/"

# Global constants describing the data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = steer_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = steer_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    We want variables pinned to the CPU so that they can be accessed if using multiple GPUs
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    # TODO assign to core based on load
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def distorted_inputs(data_dirs=DATA_DIRS, batch_size=BATCH_SIZE):
    """Construct distorted input for training using the Reader ops.
    Returns:
      images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    images, labels = steer_input.distorted_inputs(data_dirs, batch_size)
    return images, labels


def inputs(data_dir=EVAL_DATA_DIR, batch_size=BATCH_SIZE):
    """Construct input for evaluation using the Reader ops.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    images, labels = steer_input.inputs(data_dir, batch_size)
    return images, labels


def inference(images):
    """Build the steering model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Returns:
        Network steering predictions
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 24], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding="VALID")
        biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 24, 36],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [36], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 36, 48],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [48], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 48, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)

    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)

    # local1
    with tf.variable_scope('local1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(conv5, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 100],
                                              stddev=0.04, wd=0.0)
        biases = _variable_on_cpu('biases', [100], tf.constant_initializer(0.1))
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local2
    with tf.variable_scope('local2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[100, 50],
                                              stddev=0.04, wd=0.0)
        biases = _variable_on_cpu('biases', [50], tf.constant_initializer(0.1))
        local2 = tf.nn.relu(tf.matmul(local1, weights) + biases, name=scope.name)

    # local3
    with tf.variable_scope('local3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[50, 10],
                                              stddev=0.04, wd=0.0)
        biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases, name=scope.name)

    # Predict
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[10, 1], stddev=0.04, wd=0.0)
        local4 = tf.matmul(local3, weights, name=scope.name)

    return local4


def loss(outputs, targets):
    """
    Calculate L2 loss
    """
    final_loss = tf.reduce_mean(tf.square(tf.subtract(outputs, targets)))
    tf.losses.add_loss(final_loss) # We use a custom loss function – unless we add it, it won't be in collection of losses
    return final_loss

def train(total_loss, global_step):  # Simplify considerably to fit new architecture
    """Train model:
    Create an optimizer and apply to all trainable variables. Add moving average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Compute gradients.
    opt = tf.train.MomentumOptimizer(lr,0.1,use_nesterov=True)
    grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op
