# HRNet
### Input

<img src="test.png" width="640px">

(from https://www.cityscapes-dataset.com/)

Ailia input shape: (512, 1024, 3) RGB order Range: [0, 255] 8-bit unsigned integer

### Output
<img src="result.png" width="640px">


### Usage
Automatically downloads the tflite file on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image, 
`$ python3 hrnet_segmentation.py`

If you want to specify the input image, put the image path after the --input option.
You can use --savepath option to change the name of the output file to save.

`$ python3 hrnet_segmentation.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH`

By adding the --video option, you can input the video.
If you pass 0 as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

`$ python3 hrnet_segmentation.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH`

We have three pretrained-model:
  - HRNetV2-W48
  - HRNetV2-W18-Small-v1
  - HRNetV2-W18-Small-v2 (default) 

You can specify the architecture you want following --arch / -a option.

`$ python3 hrnet_segmentation.py --arch HRNetV2-W48`

Two versions of the model are provided: full integer quantization (8-bit) and full precision floating point (32-bit). 
By default, the full integer quantization is used but the user can select the other version by passing the --float flag.

`$ python3 hrnet_segmentation.py --float`


### Reference
[https://github.com/HRNet/HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)

### Framework
Tensorflow 2.11.0

### Netron
- [HRNetV2-W48_integer_quant](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/hrnet/HRNetV2-W48_integer_quant.tflite)
- [HRNetV2-W48](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/hrnet/HRNetV2-W48.tflite)
- [HRNetV2-W18-Small-v1_integer_quant](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/hrnet/HRNetV2-W18-Small-v1_integer_quant.tflite)
- [HRNetV2-W18-Small-v1](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/hrnet/HRNetV2-W18-Small-v1.tflite)
- [HRNetV2-W18-Small-v2_integer_quant](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/hrnet/HRNetV2-W18-Small-v2_integer_quant.tflite)
- [HRNetV2-W18-Small-v2](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/hrnet/HRNetV2-W18-Small-v2.tflite)
