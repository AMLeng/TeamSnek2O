import queue
import threading
import pygame
import numpy as np
import cv2
import time
import os
import shutil
import platform

# Deprecated on Darwin and Windows because memory leak, still active on Linux because ImageGrab has no Linux support
if platform.system().startswith('Linux'):
    from mss import mss
else:
    from PIL import ImageGrab

if platform.system().startswith('Linux'):
    from screeninfo import get_monitors
elif platform.system().startswith('Darwin'):
    import AppKit
elif platform.system().startswith('Windows'):
    from win32api import GetSystemMetrics

CONTROLLER = 0
STEERING_AXIS = 0
SCREEN = 0

# Write frequency in ms
WRITE_FREQUENCY = 150

# Set to None for automatic setting
# For 3360x2100 screen
IMAGE_FRONT_BORDER_LEFT = 108
IMAGE_FRONT_BORDER_RIGHT = 3258
IMAGE_FRONT_BORDER_TOP = 400
IMAGE_FRONT_BORDER_BOTTOM = 2070

SAVE_HEIGHT = 480

DEBUG = False


class MakeFrame:
    cached_size = None
    use_mss = platform.system().startswith('Linux')

    def get_screen_bbox(self):
        '''
        Query the screen size. OS specific implementation.
        :return: width, height
        '''
        # OS checking because no good cross compatibility

        if self.cached_size is None:
            screen_id = SCREEN
            if screen_id is None:
                screen_id = 0
            if platform.platform().startswith('Linux'):
                screen_res = get_monitors()[screen_id]
                width = screen_res.width
                height = screen_res.height
            elif platform.system().startswith('Darwin'):
                screen_size = AppKit.NSScreen.screens()[SCREEN].frame().size
                width = screen_size.width
                height = screen_size.height
            elif platform.system().startswith('Windows'):
                width = GetSystemMetrics(0)
                height = GetSystemMetrics(1)
            else:
                print("Could not get screen size on unsupported OS " + platform.system() + ", defaulting to 640x480")
                width = 640
                height = 480

            if IMAGE_FRONT_BORDER_LEFT is None:
                self.cached_size[0] = height
            if IMAGE_FRONT_BORDER_RIGHT is None:
                self.cached_size[1] = height
            if IMAGE_FRONT_BORDER_TOP is None:
                self.cached_size[2] = width
            if IMAGE_FRONT_BORDER_BOTTOM is None:
                self.cached_size[3] = width

        return self.cached_size[0], self.cached_size[1], self.cached_size[2], self.cached_size[3]

    def make_frame(self):
        # Capture the whole game

        if self.use_mss:
            sct = mss()
            image_raw = sct.shot()
            frame = np.array(image_raw)
            main = frame[self.get_screen_bbox()]
        else:
            frame_raw = ImageGrab.grab(bbox=self.get_screen_bbox())
            main = np.uint8(frame_raw)
            main = cv2.cvtColor(main, cv2.COLOR_BGR2RGB)

        # frame = Image.frombytes('RGB', (IMAGE_FRONT_BORDER_TOP, IMAGE_FRONT_BORDER_LEFT), image)
        # main = frame[IMAGE_FRONT_BORDER_TOP:IMAGE_FRONT_BORDER_BOTTOM,
        #       IMAGE_FRONT_BORDER_LEFT:IMAGE_FRONT_BORDER_RIGHT]

        # gray = cv2.cvtColor(main, cv2.COLOR_BGR2GRAY)
        # blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # edges = cv2.Canny(blur_gray, 50, 150)
        # dilated = cv2.dilate(edges, (3,3), iterations=2)

        # Resize image to save some space (height = 100px)
        ratio = main.shape[1] / main.shape[0]
        return cv2.resize(main, (round(ratio * SAVE_HEIGHT), SAVE_HEIGHT))


class RecordingThread(threading.Thread):
    lock = threading.Lock()
    running = True

    image_queue = queue.Queue(maxsize=0)

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        with RecordingThread.lock:
            RecordingThread.running = True

        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(CONTROLLER)
        self.joystick.init()

        if not os.path.exists("captured/"):
            os.mkdir("captured")

    def current_milli_time(self):
        return int(round(time.time() * 1000))

    def stop(self):
        with RecordingThread.lock:
            RecordingThread.running = False

    def run(self):

        timestamp = 0

        framemaker = MakeFrame()

        while RecordingThread.running:
            if self.current_milli_time() - timestamp > WRITE_FREQUENCY:

                frame = framemaker.make_frame()

                pygame.event.pump()
                # Capture the whole game

                if DEBUG:
                    print(pygame.event.get())
                axis = self.joystick.get_axis(STEERING_AXIS)  # -180 to 180 "degrees"

                # Save frame every 150ms
                timestamp = self.current_milli_time()
                self.image_queue.put((frame, timestamp, axis))

    def get_frame(self):
        """
        Gets the oldest frame from the queue.
        :return: frame, timestamp, axis
        """
        return self.image_queue.get()


class WritingThread(threading.Thread):
    # Make sure to include the trailing slash
    OUT_FOLDER = 'data/eurotruck/'

    lock = threading.Lock()
    running = True

    def __init__(self, recorder):
        # move up to the root directory of the project
        os.chdir("..")

        # delete old files
        if os.path.exists(self.OUT_FOLDER):
            print("Deleting old data")
            time.sleep(5)
            shutil.rmtree(self.OUT_FOLDER)
            print("Deleted old data")

        # make the necessary directories
        os.makedirs(self.OUT_FOLDER + 'data/')

        self.log = open(self.OUT_FOLDER + 'approximatedTimestamps.txt', 'w')
        self.recorder = recorder

        threading.Thread.__init__(self, daemon=True)
        with WritingThread.lock:
            WritingThread.running = True

    def stop(self):
        with WritingThread.lock:
            WritingThread.running = False
        self.log.close()
        self.recorder.stop()

    def run(self):
        self.recorder.start()

        while WritingThread.running:
            frame, timestamp, axis = self.recorder.get_frame()
            name = 'data/' + str(timestamp) + '.jpg'
            cv2.imwrite(self.OUT_FOLDER + name, frame)
            if DEBUG:
                print('Wrote ' + self.OUT_FOLDER + name)
            self.log.write(name + ' ' + str(axis) + '\n')
            self.log.flush()


def main():
    pass


if __name__ == "__main__":
    rt = RecordingThread()
    wt = WritingThread(rt)
    wt.start()
    wt.join()
