# MobileNet

### Input

<img src="clock.jpg" width="320px">

Ailia input shape: (224, 224, 3)  
Range: [0, 255] 8-bit unsigned integer

### Output
```
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=235
+ idx=1
  category=892[wall clock ]
  prob=21
+ idx=2
  category=999[toilet tissue, toilet paper, bathroom tissue ]
  prob=0
```

### Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 mobilenetv1.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 mobilenetv1.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 mobilenetv1.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```


### Reference

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)


### Framework
TensorFlow 2.4.1, 1.15

### Netron

- [mobilenetv1_quant_recalib.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/mobilenetv1/mobilenetv1_quant_recalib.tflite)
- [mobilenetv1_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/mobilenetv1/mobilenetv1_quant.tflite)
- [mobilenetv1_float.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/mobilenetv1/mobilenetv1_float.tflite)
