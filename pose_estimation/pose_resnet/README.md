# Simple Baselines for Human Pose Estimation and Tracking

## Input

![Input](input.jpg)

(Image from https://pixabay.com/ja/photos/%E5%A5%B3%E3%81%AE%E5%AD%90-%E7%BE%8E%E3%81%97%E3%81%84-%E8%8B%A5%E3%81%84-%E3%83%9B%E3%83%AF%E3%82%A4%E3%83%88-5204299/)

Ailia input shape: (1, 3, 256, 192)  
Range: [-2.0, 2.0]

## Output

![Output](output.png)

## Usage

Automatically downloads the tflite files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 pose_resnet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 pose_resnet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 pose_resnet.py --video VIDEO_PATH
```

Two versions of the model are provided: full integer quantization (8-bit) and full precision floating point (32-bit). 
By default, the full integer quantization is used but the user can select the other version by passing the --float flag.
```bash
$ python3 midas.py --float
```

## Reference

[Simple Baselines for Human Pose Estimation and Tracking](https://github.com/microsoft/human-pose-estimation.pytorch)

## Framework

tensorflow 2.12.0

## Netron

[pose_resnet_50_256x192_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pose_resnet/pose_resnet_50_256x192_float32.tflite)

[pose_resnet_50_256x192_int8.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pose_resnet/pose_resnet_50_256x192_int8.tflite)

