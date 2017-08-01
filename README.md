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
- [screeninfo](https://pypi.python.org/pypi/screeninfo)

#### Executables
`steer_train.py` and `steer_eval.py` are the two main executables, with self-evident names to match. `steer_train.py` should work out of the box, while `steer_eval.py` requires a saved checkpoint (`.ckpt`) file to load in variable values (saved automatically by `steer_train.py`.)

#### Useful constants
- `steer_train.py`
  - `STEPS_TO_TRAIN`: How many steps to go through before exiting automatically. 
  - `LOG_RATE`: How often to save out variable values. Default 1.
- `steer_input.py`
  - `NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN`: Self evident. Set as high as possible without exceeding RAM.
  - `NUM_EXAMPLES_PER_EPOCH_FOR_EVAL`: Self evident. Set as high as possible without exceeding RAM.
 
#### Running
`steer_train.py` can be passed an optional command-line argument with the flag `--save` to attempt to load from a previous checkpoint, with an optional path to specify the directory of the checkpoint.
