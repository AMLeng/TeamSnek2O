import queue
import threading
import pygame
import numpy as np
import cv2
import time
import os
# Deprecated because memory leak
# from mss import mss
from PIL import ImageGrab
import shutil
import platform

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
WRITE_FREQUENCY = 50

IMAGE_FRONT_BORDER_LEFT = 240
IMAGE_FRONT_BORDER_RIGHT = 2640
IMAGE_FRONT_BORDER_TOP = 0
IMAGE_FRONT_BORDER_BOTTOM = 1800

SAVE_HEIGHT = 480


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

    def get_screen_bbox(self):
        '''
        Query the screen size. OS specific implementation.
        :return: width, height
        '''
        # OS checking because no good cross compatibility
        screen_id = SCREEN
        if screen_id is None:
            screen_id = 0
        if platform.platform().startswith('Linux'):
            screen_res = get_monitors()[screen_id]
            return screen_res.width, screen_res.height
        elif platform.system().startswith('Darwin'):
            screen_size = AppKit.NSScreen.screens()[SCREEN].frame().size
            return screen_size.width, screen_size.height
        elif platform.system().startswith('Windows'):
            return GetSystemMetrics(0), GetSystemMetrics(1)
        else:
            print("Could not get screen size on unsupported OS " + platform.system() + ", defaulting to 640x480")
            return 640, 480

    def stop(self):
        with RecordingThread.lock:
            RecordingThread.running = False

    def run(self):

        timestamp = 0

        while RecordingThread.running:
            if self.current_milli_time() - timestamp > WRITE_FREQUENCY:
                pygame.event.pump()
                # Capture the whole game

                '''
                sct = mss()
                width, height = self.get_screen_bbox()
                image_raw = sct.grab({'top': 0, 'left': 0, 'width': width, 'height': height})
                image = np.array(image_raw)
                frame = image
                '''

                frame_raw = ImageGrab.grab()
                frame = np.uint8(frame_raw)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # frame = Image.frombytes('RGB', (IMAGE_FRONT_BORDER_TOP, IMAGE_FRONT_BORDER_LEFT), image)
                main = frame[IMAGE_FRONT_BORDER_TOP:IMAGE_FRONT_BORDER_BOTTOM,
                       IMAGE_FRONT_BORDER_LEFT:IMAGE_FRONT_BORDER_RIGHT]

                # gray = cv2.cvtColor(main, cv2.COLOR_BGR2GRAY)
                # blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)
                # edges = cv2.Canny(blur_gray, 50, 150)
                # dilated = cv2.dilate(edges, (3,3), iterations=2)

                # Resize image to save some space (height = 100px)
                ratio = main.shape[1] / main.shape[0]
                resized = cv2.resize(main, (round(ratio * SAVE_HEIGHT), SAVE_HEIGHT))

                # cv2.imshow('cap', dilated)
                # cv2.imshow('resized', resized)

                print(pygame.event.get())
                axis = self.joystick.get_axis(STEERING_AXIS) * 180  # -180 to 180 "degrees"

                # Save frame every 150ms
                timestamp = self.current_milli_time()
                self.image_queue.put((resized, timestamp, axis))

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
