# MobileNetV2

### Input

<img src="clock.jpg" width="320px">

Ailia input shape: (224, 224, 3)  
Range: [0, 255] 8-bit unsigned integer

### Output
```
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=114
+ idx=1
  category=892[wall clock ]
  prob=56
+ idx=2
  category=826[stopwatch, stop watch ]
  prob=8
```

### Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 mobilenetv2.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 mobilenetv2.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 mobilenetv2.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```


### Reference

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)


### Framework
TensorFlow 2.4.1, 1.15

### Netron

- [mobilenetv2_quant_recalib.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/mobilenetv2/mobilenetv2_quant_recalib.tflite)
- [mobilenetv2_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/mobilenetv2/mobilenetv2_quant.tflite)
- [mobilenetv2_float.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/mobilenetv2/mobilenetv2_float.tflite)
