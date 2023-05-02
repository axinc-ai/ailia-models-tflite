#### definition of constant and data structure ####

from collections import namedtuple

#### definition of constant ####
MULTITHREAD_AUTO = 0
ENVIRONMENT_AUTO = -1

PROFILE_DISABLE = 0
PROFILE_AVERAGE = 1


IMAGE_FORMAT_RGBA = 0x00
IMAGE_FORMAT_BGRA = 0x01
IMAGE_FORMAT_RGB = 0x02
IMAGE_FORMAT_BGR = 0x03


IMAGE_FORMAT_RGBA_B2T = 0x10
IMAGE_FORMAT_BGRA_B2T = 0x11


NETWORK_IMAGE_FORMAT_BGR = 0
NETWORK_IMAGE_FORMAT_RGB = 1
NETWORK_IMAGE_FORMAT_GRAY = 2
NETWORK_IMAGE_FORMAT_GRAY_EQUALIZE = 3

NETWORK_IMAGE_CHANNEL_FIRST = 0
NETWORK_IMAGE_CHANNEL_LAST = 1

NETWORK_IMAGE_RANGE_U_INT8 = 0  # 0 .. 255
NETWORK_IMAGE_RANGE_S_INT8 = 1  # -128 .. 127
NETWORK_IMAGE_RANGE_U_FP32 = 2  # 0.0 .. 1.0
NETWORK_IMAGE_RANGE_S_FP32 = 3  # -1.0 .. 1.0
NETWORK_IMAGE_RANGE_IMAGENET = 4  # ImageNet mean&std normalization

DETECTOR_ALGORITHM_YOLOV1 = 0
DETECTOR_ALGORITHM_YOLOV2 = 1
DETECTOR_ALGORITHM_YOLOV3 = 2
DETECTOR_ALGORITHM_YOLOV4 = 3
DETECTOR_ALGORITHM_YOLOX = 4
DETECTOR_ALGORITHM_SSD = 8

DETECTOR_FLAG_NORMAL = 0

POSE_ALGORITHM_ACCULUS_POSE = (0)
POSE_ALGORITHM_ACCULUS_FACE = (1)
POSE_ALGORITHM_ACCULUS_UPPOSE = (2)
POSE_ALGORITHM_ACCULUS_UPPOSE_FPGA = (3)
POSE_ALGORITHM_ACCULUS_HAND = (5)
POSE_ALGORITHM_OPEN_POSE = (10)
POSE_ALGORITHM_LW_HUMAN_POSE = (11)
POSE_ALGORITHM_OPEN_POSE_SINGLE_SCALE = (12)

POSE_KEYPOINT_NOSE = (0)
POSE_KEYPOINT_EYE_LEFT = (1)
POSE_KEYPOINT_EYE_RIGHT = (2)
POSE_KEYPOINT_EAR_LEFT = (3)
POSE_KEYPOINT_EAR_RIGHT = (4)
POSE_KEYPOINT_SHOULDER_LEFT = (5)
POSE_KEYPOINT_SHOULDER_RIGHT = (6)
POSE_KEYPOINT_ELBOW_LEFT = (7)
POSE_KEYPOINT_ELBOW_RIGHT = (8)
POSE_KEYPOINT_WRIST_LEFT = (9)
POSE_KEYPOINT_WRIST_RIGHT = (10)
POSE_KEYPOINT_HIP_LEFT = (11)
POSE_KEYPOINT_HIP_RIGHT = (12)
POSE_KEYPOINT_KNEE_LEFT = (13)
POSE_KEYPOINT_KNEE_RIGHT = (14)
POSE_KEYPOINT_ANKLE_LEFT = (15)
POSE_KEYPOINT_ANKLE_RIGHT = (16)
POSE_KEYPOINT_SHOULDER_CENTER = (17)
POSE_KEYPOINT_BODY_CENTER = (18)

POSE_UPPOSE_KEYPOINT_NOSE = (0)
POSE_UPPOSE_KEYPOINT_EYE_LEFT = (1)
POSE_UPPOSE_KEYPOINT_EYE_RIGHT = (2)
POSE_UPPOSE_KEYPOINT_EAR_LEFT = (3)
POSE_UPPOSE_KEYPOINT_EAR_RIGHT = (4)
POSE_UPPOSE_KEYPOINT_SHOULDER_LEFT = (5)
POSE_UPPOSE_KEYPOINT_SHOULDER_RIGHT = (6)
POSE_UPPOSE_KEYPOINT_ELBOW_LEFT = (7)
POSE_UPPOSE_KEYPOINT_ELBOW_RIGHT = (8)
POSE_UPPOSE_KEYPOINT_WRIST_LEFT = (9)
POSE_UPPOSE_KEYPOINT_WRIST_RIGHT = (10)
POSE_UPPOSE_KEYPOINT_HIP_LEFT = (11)
POSE_UPPOSE_KEYPOINT_HIP_RIGHT = (12)
POSE_UPPOSE_KEYPOINT_SHOULDER_CENTER = (13)
POSE_UPPOSE_KEYPOINT_BODY_CENTER = (14)

POSE_KEYPOINT_CNT = (19)
POSE_UPPOSE_KEYPOINT_CNT = (15)
POSE_HAND_KEYPOINT_CNT = (21)

#### definition of data structure ####
Environment = namedtuple("Environment", ["id", "type", "name", "backend", "props"])
DetectorObject = namedtuple("DetectorObject", ["category", "prob", "x", "y", "w", "h"])
ClassifierClass = namedtuple("ClassifierClass", ["category", "prob"])
PoseEstimatorKeypoint = namedtuple("PoseEstimatorKeypoint", ["x", "y", "z_local", "score", "interpolated"])
PoseEstimatorObjectPose = namedtuple("PoseEstimatorObjectPose", ["points", "total_score", "num_valid_points", "id", "angle_x", "angle_y", "angle_z"])
PoseEstimatorObjectUpPose = namedtuple("PoseEstimatorObjectUpPose", ["points", "total_score", "num_valid_points", "id", "angle_x", "angle_y", "angle_z"])
PoseEstimatorObjectHand = namedtuple("PoseEstimatorObjectHand", ["points", "total_score"])

### definition of data type for Structured array in NumPy ###
NumpyDetectorRectangle = [("x", "f4"), ("y", "f4"), ("w", "f4"), ("h", "f4")]
NumpyDetectorObject = [("category", "i4"), ("prob", "f4"), ("box", NumpyDetectorRectangle)]
NumpyClassifierClass = [("category", "i4"), ("prob", "f4")]
NumpyPoseEstimatorKeypoint = [("x", "f4"), ("y", "f4"), ("z_local", "f4"), ("score", "f4"), ("interpolated", "i4")]
NumpyPoseEstimatorObjectPose = [("points", NumpyPoseEstimatorKeypoint, POSE_KEYPOINT_CNT), ("total_score", "f4"), ("num_valid_points", "i4"), ("id", "i4"), ("angle_x", "f4"), ("angle_y", "f4"), ("angle_z", "f4")]
NumpyPoseEstimatorObjectUpPose = [("points", NumpyPoseEstimatorKeypoint, POSE_UPPOSE_KEYPOINT_CNT), ("total_score", "f4"), ("num_valid_points", "i4"), ("id", "i4"), ("angle_x", "f4"), ("angle_y", "f4"), ("angle_z", "f4")]
NumpyPoseEstimatorObjectHand = [("points", NumpyPoseEstimatorKeypoint, POSE_HAND_KEYPOINT_CNT), ("total_score", "f4")]
