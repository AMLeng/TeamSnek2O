# Team 🐍<sub>2</sub>O

![No step on snek](https://i.imgur.com/c4EYfd9.png)

### A program for driving cars. Hopefully.

This program is based on a modified version of the [TensorFlow CIFAR-10 CNN tutorial.](http://tensorflow.org/tutorials/deep_cnn/)


## Instructions

#### Dependencies
**All**
- [TensorFlow](https://www.tensorflow.org)

**Eurotruck Simulator 2 hooks**
- [OpenCV Python bindings](http://docs.opencv.org/3.2.0/d6/d00/tutorial_py_root.html)
- [Pygame](https://www.pygame.org/news)
- [Numpy](https://github.com/numpy/numpy)
- **Linux**
  - [mss](https://pypi.python.org/pypi/mss/)
  - [python-uinput](https://github.com/tuomasjjrasanen/python-uinput)
  - [screeninfo](https://pypi.python.org/pypi/screeninfo) (recommended)
- **macOS**
  - [PIL](http://www.pythonware.com/products/pil/)
  - [PyObjC/AppKit](https://pythonhosted.org/pyobjc/)
  - [foohid](https://github.com/unbit/foohid-py)
    - Make sure to install the latest version from GitHub, `pip3` doesn't contain the latest release 0.2.
- **Windows**
  - [PIL](http://www.pythonware.com/products/pil/)
  - [vJoy](http://vjoystick.sourceforge.net/site/)
  - [pywin32](https://sourceforge.net/projects/pywin32/) (recommended)

#### Executables
`steer_train.py` and `steer_eval.py` are the two main executables, with self-evident names to match. `steer_train.py` should work out of the box, while `steer_eval.py` requires a saved checkpoint (`.ckpt`) file to load in variable values (saved automatically by `steer_train.py`.)
If using multiple GPUs, use `steer_train_multi.py`.

To capture data from ETS2, use `eurotruck_processing.py`. 

#### Useful constants
- `steer_train.py`
  - `STEPS_TO_TRAIN`: How many steps to go through before exiting automatically. 
  - `LOG_RATE`: How often to save out variable values. Default 1.
- `steer_input.py`
  - `NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN`: Self evident. Set as high as possible without exceeding RAM.
  - `NUM_EXAMPLES_PER_EPOCH_FOR_EVAL`: Self evident. Set as high as possible without exceeding RAM.
- `eurotruck_processing.py`
  - `CONTROLLER`: Which controller to capture steering data from. 0 is usually right.
    - If you need help figuring out which one it is, use [jstest-gtk](https://github.com/Grumbel/jstest-gtk) (Linux), [Controllers Lite](https://itunes.apple.com/us/app/controllers-lite/id673660806?mt=12) (Darwin), or [the system Game Controllers utility](https://support.xbox.com/en-US/xbox-on-windows/accessories/calibrate-xbox-360-controller-for-windows) (Windows.)
  - `STEERING_AXIS`: Which axis on the controller to capture data from.
  - `SCREEN`: The screen to capture data from. If you've got only one screen, 0 is correct.
  - `WRITE_FREQUENCY`: How often to capture data. Experimental testing has shown this value doesn't always work properly, and data is often captured more slowly.
  - `IMAGE_FRONT_BORDER_[]`: The *absolute* coordinates of the [] side of the image
    - e.g. for a 1440x1080 image centered on a 1920x1080 screen: `LEFT = 240`, `RIGHT = 1680`, `TOP = 0`, `BOTTOM = 0`
  - `SAVE_HEIGHT`: What height to scale the image down to. Aspect ratio is maintained.
 
#### Running
- `steer_train.py` can be passed an optional command-line argument with the flag `--save` to attempt to load from a previous checkpoint, with an optional path to specify the directory of the checkpoint.

#### Troubleshooting

- **macOS**
  - `Symbol not found: _PyString_FromString` when trying to run ETS2
    - foohid-py isn't updated to the latest version 0.2, and is probably getting 2.7 and 3.x confused. Install using `sudo pip3 install git+git://github.com/unbit/foohid-py.git
 ` instead of `sudo pip3 install foodhid`
  - `unable to open it_unbit_foohid service` when trying to run ETS2 
    - Bug in foodhid. Try restarting your computer. If that doesn't work, run `sudo kextunload /Library/Extensions/foohid.kext;sudo rm -rf /System/Library/Extensions/foohid.kext` then reinstall foodhid. If you encounter any errors, try restarting. Kexts are fun like that.