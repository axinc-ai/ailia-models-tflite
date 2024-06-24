# Vision Transformer

### Input

<img src="daisy.jpg" width="320px">

Ailia input shape: (224, 224, 3)  
Range: [-128, 127] 8-bit signed integer

### Output
```
TBD
```

### Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 vision_transformer.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 vision_transformer.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 vision_transformer.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```


### Reference

[Vision Transformer in TensorFlow 2.x](https://github.com/hrithickcodes/vision_transformer_tf)


### Framework
TensorFlow 2.6

### Netron

- [vision_transformer_float.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/vision_transformer/vision_transformer_float.tflite)
