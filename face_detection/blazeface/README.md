# BlazeFace

### Input

<img src="input.png" width="320px">

(Image from https://github.com/hollance/BlazeFace-PyTorch/blob/master/3faces.png)

Ailia input shape: (128, 128, 3)  
Range: [-1, 1]

### Output

<img src="result.png" width="320px">

### Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 blazeface.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 blazeface.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 blazeface.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

Two versions of the model are provided: full integer quantization (8-bit) and
full precision floating point (32-bit). By default, the full integer
quantization is used but the user can select the other version by passing the
`--float` flag.
```bash
$ python3 blazeface.py --float
```

### Reference

[PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/)


### Framework
TensorFlow 2.4.1, 1.15

### Netron

- [face_detection_front.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/blazeface/face_detection_front.tflite)
- [face_detection_front_128_full_integer_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/blazeface/face_detection_front_128_full_integer_quant.tflite)
