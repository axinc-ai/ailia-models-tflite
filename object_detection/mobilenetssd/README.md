# MobileNetV2-SSDLite

### Input

<img src="couple.jpg" width="320px">

Ailia input shape: (300, 300, 3) RGB order
Range: [-1, 1]

### Output
<img src="output.png" width="320px">

### Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 mobilenetv2ssdlite.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 mobilenetv2ssdlite.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 mobilenetv2ssdlite.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```


### Reference

[PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/master/006_mobilenetv2-ssdlite/01_coco/03_integer_quantization)


### Framework
TensorFlow 2.4.1, 1.15

### Netron

[ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/mobilenetssd/ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess.tflite)
