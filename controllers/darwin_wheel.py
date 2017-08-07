import foohid
import threading
import struct


class DarwinWheel(threading.Thread):
    angle = 0

    lock = threading.Lock()
    running = True

    def __init__(self):

        threading.Thread.__init__(self, daemon=True)
        with DarwinWheel.lock:
            DarwinWheel.running = True

        # Because random reasons
        joypad = (
            0x05, 0x01,
            0x09, 0x05,
            0xa1, 0x01,
            0xa1, 0x00,
            0x05, 0x09,
            0x19, 0x01,
            0x29, 0x10,
            0x15, 0x00,
            0x25, 0x01,
            0x95, 0x10,
            0x75, 0x01,
            0x81, 0x02,
            0x05, 0x01,
            0x09, 0x30,
            0x09, 0x31,
            0x09, 0x32,
            0x09, 0x33,
            0x15, 0x81,
            0x25, 0x7f,
            0x75, 0x08,
            0x95, 0x04,
            0x81, 0x02,
            0xc0,
            0xc0)
        foohid.create("FooHID joypad", struct.pack('{0}B'.format(len(joypad)), *joypad), "SN 123", 2, 3)

    def stop(self):
        with DarwinWheel.lock:
            DarwinWheel.running = False
            foohid.destroy("FooHID joypad")

    # Don't think anything needs to happen here
    def run(self):
        pass

    def set_angle(self, angle):
        if DarwinWheel.running:
            # Translate from -1-1 to 0-255
            step = int(angle * 127.5 + 127.5)
            print(step)
            foohid.send("FooHID joypad", struct.pack('H4B', 0, step, 0, 0, 0))