# i.MX8 Support Status

Ailia TFLite Runtime version 1.1.12 and later now supports the Python API on i.MX8. The models that have been confirmed to work are listed below. In the future, we plan to expand the range of compatible models by supporting the C API and offloading CPU operators that are not supported by the NPU.

|Model|Supported|
|-----|-----|
|u2net|OK|
|midas|NG (int object has no attribute transpose)|
|blazeface|OK|
|facemesh|OK|
|face_classification|OK|
|blazehand|OK|
|mobilenet|NG (reshape failed)|
|mobilenetv2|OK|
|resnet50|OK|
|efficientnetlite|OK|
|squeezenet|OK|
|vgg16|OK|
|googlenet|OK|
|deeplabv3+|OK|
|hrnet_segmentation|OK|
|yolov3_tiny|NG (Invalid result)|
|yolox_tiny|OK|
|efficientdetlite|OK|
|pose_resnet|NG (Didn't find op for builtin opcode 'TRANSPOSE_CONV' version '4')|
|espcn|OK|
|srresnet|NG (Didn't find op for builtin opcode 'TRANSPOSE' version' 6')|
