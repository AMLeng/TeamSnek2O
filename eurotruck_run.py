import tensorflow as tf
import threading
import sys

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

    def stop(self):
        with WritingThread.lock:
            WritingThread.running = False

    def run(self):

        while WritingThread.running:
            frame = self.framemaker.make_frame()
            image_tensor = tf.image.decode_image(frame)

            with tf.Graph().as_default():

                # Get images and labels
                # Force input pipeline to CPU:0
                #   with tf.device('/cpu:0'):
                # images, labels = steer.inputs()

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

                print(sess.run())

if __name__ == "__main__":
    wt = WritingThread(MakeFrames())
    wt.start()
    wt.join()