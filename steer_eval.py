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
# Modifications finished? (Except for renaming import)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from datetime import datetime
import time
import sys
import tensorflow as tf
import steer

EVAL_DIR = "tmp/steering_eval"


def eval():
    """Train for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = steer.distorted_inputs(EVAL_DIR)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = steer.inference(images)

        # Calculate loss.
        loss = steer.loss(logits, labels)

        print(str(datetime.now()) + " " + tf.Session().run(loss))


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(EVAL_DIR):
        tf.gfile.DeleteRecursively(EVAL_DIR)
    tf.gfile.MakeDirs(EVAL_DIR)
    eval()


if __name__ == '__main__':
    tf.app.run()
