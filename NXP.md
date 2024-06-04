# i.MX8 Support Status

Ailia TFLite Runtime version 1.1.12 and later now supports the Python API on i.MX8. The models that have been confirmed to work are listed below. In the future, we plan to expand the range of compatible models by supporting the C API and offloading CPU operators that are not supported by the NPU.

|Model|6.1.22 (2.10.0)|6.6.3 (2.14.0)|
|-----|-----|-----|
|u2net|OK|OK|
|midas|NG (int object has no attribute transpose)|NG ('int' object has no attribute 'transpose')|
|facemesh|OK|OK|
|face_classification|OK|OK|
|blazehand|OK|OK|
|mobilenet|NG (reshape failed)|NG (Dynamic input shape is not supported in reshape.)|
|mobilenetv2|OK|OK|
|resnet50|OK|OK|
|efficientnetlite|OK|OK|
|squeezenet|OK|OK|
|vgg16|OK|OK|
|googlenet|OK|OK|
|deeplabv3+|OK|OK|
|hrnet_segmentation|OK|OK|
|yolov3_tiny|NG (Invalid result)|OK|
|yolox_tiny|OK|OK|
|efficientdetlite|OK|OK|
|pose_resnet|NG (Didn't find op for builtin opcode 'TRANSPOSE_CONV' version '4')|OK|
|espcn|OK|OK|
|srresnet|NG (Didn't find op for builtin opcode 'TRANSPOSE' version' 6')|OK|
