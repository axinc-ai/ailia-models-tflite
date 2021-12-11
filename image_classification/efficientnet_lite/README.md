# EfficientNetLite

### Input

<img src="clock.jpg" width="320px">

Ailia input shape: (224, 224, 3)  
Range: [-128, 127] 8-bit signed integer

### Output
```
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=5
+ idx=1
  category=892[wall clock ]
  prob=-52
+ idx=2
  category=826[stopwatch, stop watch ]
  prob=-119
Script finished successfully.
```

### Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 efficientnet_lite.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 efficientnet_lite.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 efficientnet_lite.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```


### Reference

[efficientnet-lite-keras](https://github.com/sebastian-sz/efficientnet-lite-keras)


### Framework
TensorFlow 2.6

### Netron

[efficientnetliteb0_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/efficientnet_lite/efficientnetliteb0_quant.tflite)
