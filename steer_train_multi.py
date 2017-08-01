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
# Modifications finished? (Except for renaming import)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import sys
import argparse
import numpy as np
import tensorflow as tf
import steer

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_STEPS_PER_EPOCH_FOR_TRAIN=(int) (steer.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/steer.BATCH_SIZE)
STEPS_TO_TRAIN = 120 #This is the STEPS to train, not epochs. The epochs is given by images per train epoch (see the steer_input file), divided by steps_to_train*128
LOG_RATE = NUM_STEPS_PER_EPOCH_FOR_TRAIN #This is also in terms of steps, not epochs. If set to num_steps_per_epoch_for_train, logs once an epoch
TRAINING_DIR = "tmp/steering_train"
GPU_NAME="GPU_"
NUM_GPUS=1

def tower_loss(scope, images, labels):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 3].
        labels: Labels. 1D tensor of shape [batch_size].
    Returns:
         Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    logits = steer.inference(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = steer.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        def replace_none_with_zero(l):
            #Some gradients might be 0s, which returns none, so we have to fix that
            #https://github.com/tensorflow/tensorflow/issues/783
            return 0.0 if l==None else l
        def replace_none_with_zero(l,x):
            #Same as above, but allows the specification of a tensor shape---the same shape as x will be used
            return tf.zeros_like(x) if l==None else l
        for g, x in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(replace_none_with_zero(g,x), 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(path_to_save):
    """Train for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (steer.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 steer.BATCH_SIZE)
        decay_steps = int(num_batches_per_epoch * steer.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(steer.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        steer.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.MomentumOptimizer(lr,0.1,use_nesterov=True)

        # Get images and labels
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = steer.distorted_inputs()
        print("Images read")
        sys.stdout.flush()


        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                    [images, labels], capacity=2 * NUM_GPUS)
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(NUM_GPUS):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (GPU_NAME, i)) as scope:
                        # Dequeues one batch for the GPU
                        image_batch, label_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, image_batch, label_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False))
        init=tf.global_variables_initializer()
        sess.run(init)
        print("Beginning Training")
        # Restore from the save if applicable
        if path_to_save is not None:
            ckpt = tf.train.get_checkpoint_state(path_to_save)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print("Restoring session " + ckpt.model_checkpoint_path)
                sys.stdout.flush()
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(global_step)
                sys.stdout.flush()
            else:
                print('No checkpoint file found')
                return

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        for step in range(STEPS_TO_TRAIN):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % LOG_RATE == 0:
                num_examples_per_step = steer.BATCH_SIZE * NUM_GPUS
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / NUM_GPUS

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                            'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

                # Save the model checkpoint periodically.
                checkpoint_path = os.path.join(TRAINING_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

# TODO add args
def main(*args):  # pylint: disable=unused-argument
    parser = argparse.ArgumentParser(description="Various command line args")
    parser.add_argument('--save', nargs='?', const=TRAINING_DIR, default=None, required=False,
                        help="An optional directory containing .ckpt save files to restore.",
                        metavar="/path/to/folder")
    args = parser.parse_args()
    save_path = args.save

    if save_path is None:
        print("Deleting old training sessions... (Not actually)")
        # if tf.gfile.Exists(TRAINING_DIR):
        #    tf.gfile.DeleteRecursively(TRAINING_DIR)
        #tf.gfile.MakeDirs(TRAINING_DIR)
    train(save_path)


if __name__ == '__main__':
    tf.app.run()
