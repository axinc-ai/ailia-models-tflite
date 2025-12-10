# GFP-GAN

## Input

### aligned

![Input](face_03.png)

Shape : (1, 3, 512, 512)

(Image from https://github.com/TencentARC/GFPGAN/blob/master/inputs/whole_imgs/10045.png)

## Output

### aligned

![Output](output.png)

Shape : (1, 3, 512, 512)


## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 gfpgan.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 gfpgan.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

## Reference

- [GFPGAN](https://github.com/TencentARC/GFPGAN)

## Framework

Pytorch

## Model Format

tflite

## Netron

v1.3

[gfpgan_float.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/gfpgan/gfpgan_float.tflite)
[gfpgan_int8.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/gfpgan/gfpgan_int8.tflite)
