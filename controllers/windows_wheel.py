import threading
import pyvjoy
from pyvjoy import vjoydevice

class WindowsWheel(threading.Thread):
    lock = threading.Lock()
    running = True

    joystick = vjoydevice.VJoyDevice(1)

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        with WindowsWheel.lock:
            WindowsWheel.running = True


    def stop(self):
        pass

    # Don't think anything needs to happen here
    def run(self):
        pass

    def set_angle(self, angle):
        if WindowsWheel.running:
            # translating from -1-1 to 0x0 to 0x8000
            step = int(angle * 16384 + 16384)
            self.joystick.set_axis(pyvjoy.constants.HID_USAGE_X, step)
            # For some reason running this line makes it so it doesn't update. Yup.
            # self.joystick.update()