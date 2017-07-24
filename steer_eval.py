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

"""Evaluation for CIFAR-10.
Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.
Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys

import numpy as np
import tensorflow as tf
import steer
import steer_input

EVAL_DIR='tmp/steering_eval'
CHECKPOINT_DIR='tmp/steering_train'
NUM_EXAMPLES=steer_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
def evaluate():
    with tf.Graph().as_default():

        # Get images and labels
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        # with tf.device('/cpu:0'):
        images, labels = steer.inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = steer.inference(images)
        loss_op = steer.loss(logits, labels)

        saver = tf.train.Saver()
        print("Evaluating")
        sys.stdout.flush()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print("Restoring session "+ckpt.model_checkpoint_path)
                sys.stdout.flush()
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print(global_step)
                sys.stdout.flush()
            else:
                print('No checkpoint file found')
                return


        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0    # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Calculate loss.
            print(str(datetime.now()) + " " + sess.run(loss_op))
            sys.stdout.flush()

        except Exception as e:    # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(EVAL_DIR):
        tf.gfile.DeleteRecursively(EVAL_DIR)
    tf.gfile.MakeDirs(EVAL_DIR)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
