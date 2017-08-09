import tensorflow as tf
import threading
import sys
import numpy as np
import time
import platform

import steer

from processing.eurotruck_processing import MakeFrames

CHECKPOINT_DIR = 'data/tmp/steering_train'


class WritingThread(threading.Thread):
    lock = threading.Lock()
    running = True

    def __init__(self, framemaker):
        self.framemaker = framemaker

        threading.Thread.__init__(self, daemon=True)
        with WritingThread.lock:
            WritingThread.running = True

        if platform.system().startswith('Linux'):
            from controllers.linux_wheel import LinuxWheel
            self.wheel = LinuxWheel()
        elif platform.system().startswith('Darwin'):
            from controllers.darwin_wheel import DarwinWheel
            self.wheel = DarwinWheel()
        elif platform.system().startswith('Windows'):
            from controllers.windows_wheel import WindowsWheel
            self.wheel = WindowsWheel()

        self.wheel.start()

    def stop(self):
        with WritingThread.lock:
            self.wheel.stop()
            WritingThread.running = False

    def run(self):

        first_image = self.framemaker.make_frame()

        with tf.Graph().as_default():

            shape = np.expand_dims(np.asarray(first_image), 0).shape
            image_placeholder = tf.placeholder(tf.float32, shape=(shape))

            # Get images and labels
            # Force input pipeline to CPU:0
            #   with tf.device('/cpu:0'):
            # images, labels = steer.inputs()
            steer.BATCH_SIZE = 1
            logits = steer.inference(image_placeholder)

            saver = tf.train.Saver()
            sys.stdout.flush()
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    sys.stdout.flush()
                    # Assuming model_checkpoint_path looks something like:
                    #   /my-favorite-path/cifar10_train/model.ckpt-0,
                    # extract global_step from model_checkpoint_path.
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    sys.stdout.flush()
                else:
                    print('No checkpoint file found')
                    return


                try:
                    while WritingThread.running:
                        start_time = time.time()

                        frame = self.framemaker.make_frame()
                        '''
                        image_data = np.asarray(frame)
                        image_tensor = tf.convert_to_tensor(image_data)
                        cast_tensor = tf.cast(image_tensor, tf.float32)
                        reshaped_tensor = tf.expand_dims(cast_tensor, 0)
                        '''
                        frame_data = np.expand_dims(np.asarray(frame), 0).astype(float)
                        angle = sess.run(logits, feed_dict={image_placeholder: frame_data})


                        self.wheel.set_angle(angle)
                        print(angle, 1/(time.time() - start_time))
                except Exception as e:
                    print(e)
                    self.stop()

if __name__ == "__main__":
    wt = WritingThread(MakeFrames())
    wt.start()
    wt.join()
