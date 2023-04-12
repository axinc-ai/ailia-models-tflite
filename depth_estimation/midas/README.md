# MiDaS

## Input

![Input](input.jpg)

(Image from kitti dataset http://www.cvlibs.net/datasets/kitti/raw_data.php)

Shape : (1, h, w, 3)

## Output

![Output](input_depth.png)

Shape : (1, h, w)

## Usage
Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 midas.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 midas.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 midas.py --video VIDEO_PATH
```

By adding the `-v21` option, you can use version 2.1 model.  
(default use version 2.0 model)

If you use the version 2.1 model, you can use the small model with the `--model_type small` option.
```bash
$ python3 midas.py -v21 --model_type small
```

Two versions of the model are provided: full integer quantization (8-bit) and full precision floating point (32-bit). 
By default, the full integer quantization is used but the user can select the other version by passing the --float flag.

`$ python3 midas.py --float`

## Reference

[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://github.com/intel-isl/MiDaS)

## Framework

Tensorflow 2.12.0

## Netron

[midas_quant_recalib.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/midas/midas_quant_recalib.tflite)

[midas_float.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/midas/midas_float.tflite)

[midas_v2.1_quant_recalib.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/midas/midas_v2.1_quant_recalib.tflite)

[midas_v2.1_float.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/midas/midas_v2.1_float.tflite)

[midas_v2.1_small_quant_recalib.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/midas/midas_v2.1_small_quant_recalib.tflite)

[midas_v2.1_small_float.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/midas/midas_v2.1_small_float.tflite)