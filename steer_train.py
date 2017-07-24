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

##THIS FILE HAS BEEN MODIFIED FOR THE NWAP WORKSHOP
#Modifications finished? (Except for renaming import)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import sys
import tensorflow as tf

import steer
#Batch size of 128

STEPS_TO_TRAIN=100
LOG_RATE=10
TRAINING_DIR="tmp/steering_train"


def train():
  """Train for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = steer.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = steer.inference(images)

    # Calculate loss.
    loss = steer.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = steer.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % LOG_RATE == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = LOG_RATE * 128 / duration #NOTE:Batch size defined here
          sec_per_batch = float(duration / LOG_RATE)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
          sys.stdout.flush()

    print("Beginning Training")
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=TRAINING_DIR,
        hooks=[tf.train.StopAtStepHook(num_steps=STEPS_TO_TRAIN),
               tf.train.NanTensorHook(loss),
               _LoggerHook(),
               tf.train.CheckpointSaverHook("tmp/steering_train",save_steps=1)],
        config=tf.ConfigProto(log_device_placement=False),
        save_checkpoint_secs=60) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(TRAINING_DIR):
    tf.gfile.DeleteRecursively(TRAINING_DIR)
  tf.gfile.MakeDirs(TRAINING_DIR)
  train()


if __name__ == '__main__':
  tf.app.run()
