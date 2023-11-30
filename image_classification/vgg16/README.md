# VGG16

## Input

![Input](pizza.jpg)

Shape : (1,3,224,224)

## Output

```
class_count=3
+ idx=0
  category=963[pizza, pizza pie ]
  prob=14.844779968261719
  value=47
+ idx=1
  category=927[trifle ]
  prob=12.971166610717773
  value=34
+ idx=2
  category=926[hot pot, hotpot ]
  prob=10.52105712890625
  value=17
Script finished successfully.
```

## Usage
Automatically downloads the tflite files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 vgg16.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 vgg16.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 vgg16.py --video VIDEO_PATH
```


## Reference

[Very Deep Convolutional Networks for Large-Scale Image Recognition]( https://arxiv.org/abs/1409.1556 )

[VGG16 - Torchvision]( https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html )

[Keras Applications : VGG16]( https://keras.io/applications/#vgg16 )

[keras2caffe]( https://github.com/uhfband/keras2caffe)


## Framework

TensorFlow 2.12.0

## Netron

[vgg16_pytorch_quant_recalib.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/vgg16/vgg16_pytorch_quant_recalib.tflite)

[vgg16_pytorch_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/vgg16/vgg16_pytorch_float32.tflite)
