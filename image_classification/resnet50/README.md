# ResNet50

### Input

<img src="clock.jpg" width="320px">

Ailia input shape: (224, 224, 3)  
Range: [-128, 127] 8-bit signed integer

### Output
```
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=97
+ idx=1
  category=892[wall clock ]
  prob=-107
+ idx=2
  category=426[barometer ]
  prob=-123
Script finished successfully.
```

### Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 resnet50.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 resnet50.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 resnet50.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```


### Reference

[tf.keras.applications.resnet50.ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)


### Framework
TensorFlow 2.6

### Netron

[resnet50_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/resnet50/resnet50_quant.tflite)
