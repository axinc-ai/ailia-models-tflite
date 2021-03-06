# yolov3-tiny

## Input

![Input](input.jpg)

- Shape : (1, 3, 416, 416)  
- Range : [0.0, 1.0]

## Output

![Output](output.png)

- category : [0,79]
- probablity : [0.0,1.0]
- position : x, y, w, h [0,1]

## Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 yolov3-tiny.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 yolov3-tiny.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 yolov3-tiny.py --video VIDEO_PATH
```


## Reference

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)

## Framework

Tensorflow 2.4.1

## Netron

- [yolov3-tiny-416_full_integer_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/yolov3-tiny/yolov3-tiny-416_full_integer_quant.tflite)
- [yolov3-tiny-416.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/yolov3-tiny/yolov3-tiny-416.tflite)
