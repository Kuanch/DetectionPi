# DetectionPi
Real-time face recognition on Raspberry pi.

## ONNX and model optimization
Switch tensorflow inference to ONNX since unavaliable tensorflow C++ stable API and impovring ONNX ecosystem.

## Set up Tensorflow C++ API Library (Deprecated)
This repository makes possible the usage of the TensorFlow C++ API from the outside of the TensorFlow source code folders and without the use of the Bazel build system, please refer to [Tensorflow C++ API](https://github.com/FloopCZ/tensorflow_cc).

## TODO
- [x] Tensorflow SSD-MobileNetV2 to ONNX
- [ ] Inference code
- [ ] Deploy on Raspberry pi
- [ ] YoloV4
- [ ] Model Optimization
