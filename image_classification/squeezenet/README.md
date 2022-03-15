# SqueezeNet

### Input

<img src="clock.jpg" width="320px">

Ailia input shape: (227, 227, 3)  
Range: [0, 255] 8-bit unsigned integer

### Output
```
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=5
+ idx=1
  category=892[wall clock ]
  prob=-15
+ idx=2
  category=635[magnetic compass ]
  prob=-119
```

### Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 squeezenet.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 squeezenet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 squeezenet.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```


### Reference

[keras_squeezenet2](https://github.com/daviddexter/keras_squeezenet2)


### Framework
TensorFlow 2.7.0

### Netron

[squeezenet_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/squeezenet/squeezenet_quant.tflite)
