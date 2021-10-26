# BlazeHand

## Input

<img src="person_hand.jpg" width="320px">

(Image from https://pixabay.com/photos/stop-no-photo-no-photographing-hand-565609/)

### Detector

- ailia input shape: (1, 256, 256, 3) RGB channel order
- Pixel value range: [0, 1]

### Landmark

- ailia input shape: (batch_size, 256, 256, 3) BGR channel order
- Pixel value range: [0, 1]

## Output

<img src="output.png" width="320px">

### Detector

- ailia Predict API output:
  - Classification confidences
    - Shape: (1, 2944, 1)
  - Bounding boxes and keypoints
    - Shape: (1, 2944, 18)
- With helper functions, filtered detections with keypoints can be obtained.

### Estimator

- ailia Predict API output:
  - `hand_flag`: confidence score [0, 1] of hand presence
    - Shape: (batch_size, 1, 1, 1)
  - `handedness`: classification score [0, 1] of right handedness
    - Shape: (batch_size, 1, 1, 1)
    - Estimated probability of the right handedness (and the opposite
    handedness has an estimated probability of 1 - score).
    - Handedness is determined assuming the input image is mirrored, i.e.,
    taken with a front-facing/selfie camera with images flipped horizontally.
    If it is not the case, please swap the handedness output in the application.
  - `landmarks`: 21 hand landmarks with (x, y, z) coordinates
    - Shape: (batch_size, 63) [21 3D points]
    - x and y are normalized to [0.0, 1.0] by the image width and height
    respectively. z represents the landmark depth with the depth at the wrist
    being the origin, and the smaller the value the closer the landmark is to
    the camera. The magnitude of z uses roughly the same scale as x.
- With helper functions, image coordinates of hand landmarks can be obtained.

## Usage

Automatically downloads the tflite file on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 blazehand.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 blazehand.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 blazehand.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

By adding the `--hands` option, you can decide the maximum number of tracked hands.
By default, it allows tracking up to 2 hands.
```bash
$ python3 blazehand.py --hands 3
```

## Reference

- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/032_FaceMesh/033_Hand_Detection_and_Tracking)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)

## Framework

TensorFlow 2.4.1

## Netron

- [palm_detection_builtin_256_full_integer_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/blazepalm/palm_detection_builtin_256_full_integer_quant.tflite)
- [hand_landmark_new_256x256_full_integer_quant.tflite](https://netron.app/?url=https://storage.googleapis.com/ailia-models-tflite/blazehand/hand_landmark_new_256x256_full_integer_quant.tflite)
