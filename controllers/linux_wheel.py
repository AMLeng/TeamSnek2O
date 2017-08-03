import threading
import uinput

class LinuxWheel(threading.Thread):
    angle = 0
    device = uinput.Device({uinput.ABS_X + (-180, 180, 0, 0)})

    lock = threading.Lock()
    running = True

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        with LinuxWheel.lock:
            LinuxWheel.running = True


    def stop(self):
        pass

    # Don't think anything needs to happen here
    def run(self):
        pass

    def set_angle(self, angle):
        if LinuxWheel.running:
            self.device.emit(uinput.ABS_X, angle)
            print(angle)