import threading
import uinput

class LinuxWheel(threading.Thread):
    angle = 0
    events = (
        uinput.ABS_X + (-180, 180, 0, 0),
        uinput.ABS_Y + (-180, 180, 0, 0),
        uinput.BTN_JOYSTICK,
        )
    device = uinput.Device(events)

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
