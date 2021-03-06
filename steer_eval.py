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
import sys
import numpy as np
import tensorflow as tf
import steer
import steer_input
import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EVAL_DIR = 'data/tmp/steering_eval'
CHECKPOINT_DIR = 'data/tmp/steering_train'
NUM_EXAMPLES = steer_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
BATCH_SIZE = 128


def evaluate():
    with tf.Graph().as_default():

        # Get images and labels
        # Force input pipeline to CPU:0
        #   with tf.device('/cpu:0'):
        images, labels = steer.inputs()

        # Build a graph that computes the logits predictions from the inference model.
        logits = steer.inference(images)
        # loss_op = tf.reduce_sum(tf.square(tf.subtract(logits, labels)))
        loss_op=steer.loss(logits,labels)

        saver = tf.train.Saver()
        print("Evaluating")
        sys.stdout.flush()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print("Restoring session " + ckpt.model_checkpoint_path)
                sys.stdout.flush()
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from model_checkpoint_path.
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print(global_step)
                sys.stdout.flush()
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = []

            try:
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(NUM_EXAMPLES / BATCH_SIZE))
                error_sum = 0
                step = 0
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([loss_op])
                    error_sum += np.sum(predictions)
                    step += 1
                    for x in predictions:
                        print("Step "+str(step) +": "+str(x))
                precision = error_sum / (num_iter)
                print('%s: Average Error = %.3f' % (datetime.now(), precision))
                sys.stdout.flush()

            except Exception as e:  # pylint: disable=broad-except
                print(e)
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


# TODO add args
def main(args):  # pylint: disable=unused-argument
    if tf.gfile.Exists(EVAL_DIR):
        tf.gfile.DeleteRecursively(EVAL_DIR)
    tf.gfile.MakeDirs(EVAL_DIR)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
