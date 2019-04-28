## DetectionPi
Process images and detect object on raspberry pi with tensorflow and opencv.

## Set up Tensorflow C++ API Library
This repository makes possible the usage of the TensorFlow C++ API from the outside of the TensorFlow source code folders and without the use of the Bazel build system, please refer to [Tensorflow C++ API](https://github.com/FloopCZ/tensorflow_cc).

## Usage
```
mkdir build && cd build
cmake .. && make
./tf_ex "PATH_TO_INFERED_IMAGE" "PATH_TO_SAVED_MODEL"
```
