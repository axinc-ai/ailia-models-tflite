# U^2-Net

### input
![input_image](input.png)  
(Image from https://github.com/NathanUA/U-2-Net/blob/master/test_data/test_images/girl.png)
- Ailia input shape: (1, 320, 320, 3)  

### output
![output_image](output.png)

### usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 u2net.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 u2net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 u2net.py --video VIDEO_PATH
```

You can select a pretrained model by specifying `-a large`(default) or `-a small`.

```bash
$ python3 u2net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH -a small
```

When using ailia SDK 1.2.3 or earlier, you must use a lower accurate model by specifying `--opset 10`.

```bash
$ python3 u2net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH --opset 10
```

Add the `--composite` option if you want to combine the input image with the calculated alpha value.

```bash
$ python3 u2net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH --opset 11 --composite
```

Add the `--float` option if you want to use float32 model for higher precision.

```bash
$ python3 u2net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH --opset 11 --float
```

### Reference

[U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net)


### Framework
TensorFlow 2.10.0

### Netron

量子化モデル
- [u2net_full_integer_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/u2net/u2net_full_integer_quant.tflite)
- [u2net_opset11_full_integer_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/u2net/u2net_opset11_full_integer_quant.tflite)
- [u2netp_full_integer_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/u2net/u2netp_full_integer_quant.tflite)
- [u2netp_opset11_full_integer_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/u2net/u2netp_opset11_full_integer_quant.tflite)

floatモデル
- [u2net_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/u2net/u2net_float32.tflite)
- [u2net_opset11_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/u2net/u2net_opset11_float32.tflite)
- [u2netp_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/u2net/u2netp_float32.tflite)
- [u2netp_opset11_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/u2net/u2netp_opset11_float32.tflite)
