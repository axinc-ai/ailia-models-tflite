# efficientdet lite

## Input

![Input](input.jpg)

- Shape : (1, 320, 320, 3)  
- Range : [0.0, 1.0]

## Output

![Output](output.jpg)

- category : [0,79]
- probablity : [0.0,1.0]
- position : x, y, w, h [0,1]

## Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 efficientdet_lite.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 efficientdet_lite.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 efficientdet_lite.py --video VIDEO_PATH
```


## Reference

- [EfficientDet](https://github.com/google/automl/tree/master/efficientdet)
- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/103_EfficientDet_lite)

## Framework

Tensorflow 2.7.0

## Netron

- [efficientdet_lite0_integer_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/efficientdet_lite/efficientdet_lite0_integer_quant.tflite)
- [efficientdet_lite0_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/efficientdet_lite/efficientdet_lite0_float32.tflite)
