# GoogleNet

## Input

![Input](pizza.jpg)

Ailia input shape: (224, 224, 3)  
Range : [0.0, 255.0] 

## Output

```
+ idx=0
  category=963[pizza, pizza pie ]
  prob=7.1925835609436035            
+ idx=1
  category=926[hot pot, hotpot ] 
  prob=6.819923400878906
+ idx=2
  category=567[frying pan, frypan, skillet ]
  prob=6.660032272338867
```

## Usage
Automatically downloads the tflite files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image,
```
$ python3 googlenet.py
```

If you want to specify the input image, put the image path after the `--input` option.
```
$ python3 googlenet.py --input IMAGE_PATH
```
By adding the `--video` option, you can input the video.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```
$ python3 googlenet.py --video VIDEO_PATH
```


## Reference

[Going Deeper with Convolutions]( https://arxiv.org/abs/1409.4842 )

[GOOGLENET]( https://pytorch.org/hub/pytorch_vision_googlenet/)


## Framework

TensorFlow 2.12.0

## Netron

[googlenet_float32.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/googlenet/googlenet_float32.tflite)

[googlenet_quant_recalib.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/googlenet/googlenet_quant_recalib.tflite)
