# ESPCN

## Input

![Input](lenna.png)

Ailia input shape : (1,3,64,64)  
Range : [0.0, 1.0]

## Output

![Output](output.jpg)

Ailia output shape : (1,3,192,192)  
Range : [0, 1.0]

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 espcn.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 espcn.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 espcn.py --video VIDEO_PATH
```

## Reference

[Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel/)

## Framework

TensorFlow 2.7.0

## Netron

[espcn.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/espcn/espcn.tflite)

[espcn_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/espcn/espcn_quant.tflite)

