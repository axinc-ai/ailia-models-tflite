# Real-time face detection and emotion/gender classification

## Input

![Input](lenna.png)

Input shape: (1, 64, 64, 1)  
Range: [-1.0, 1.0]

## Output

```
emotion_class_count=3
+ idx=0
  category=6 [ neutral ]
  prob=0.41015625
+ idx=1
  category=0 [ angry ]
  prob=0.1875
+ idx=2
  category=4 [ sad ]
  prob=0.1875

gender_class_count=2
+ idx=0
  category=0 [ female ]
  prob=0.94140625
+ idx=1
  category=1 [ male ]
  prob=0.0546875
```

## Usage
Automatically downloads the tflite files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 face_classification.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 face_classification.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 face_classification.py --video VIDEO_PATH
```

Two versions of the model are provided: full integer quantization (8-bit) and full precision floating point (32-bit). 
By default, the full integer quantization is used but the user can select the other version by passing the --float flag.
```bash
$ python3 face_classification.py --float
```


## Reference

[Real-time face detection and emotion/gender classification](https://github.com/oarriaga/face_classification)


## Framework

tensorflow 2.12.0


## Netron

[emotion_miniXception_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/face_classification/emotion_miniXception_float32.tflite)

[emotion_miniXception_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/face_classification/emotion_miniXception_quant.tflite)

[gender_miniXception_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/face_classification/gender_miniXception_float32.tflite)

[gender_miniXception_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/face_classification/gender_miniXception_quant.tflite)
