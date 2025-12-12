export OPTION="-b -bc 100"

cd ../
cd background_removal/u2net; python3 u2net.py ${OPTION}
cd ../../depth_estimation/midas; python3 midas.py ${OPTION}
cd ../../face_detection/blazeface; python3 blazeface.py ${OPTION}
cd ../../face_recognition/facemesh; python3 facemesh.py ${OPTION}
cd ../../hand_recognition/blazehand; python3 blazehand.py ${OPTION}
cd ../../image_classification/efficientnet_lite; python3 efficientnet_lite.py ${OPTION}
cd ../../image_classification/mobilenetv1; python3 mobilenetv1.py ${OPTION}
cd ../../image_classification/mobilenetv2; python3 mobilenetv2.py ${OPTION}
cd ../../image_classification/resnet50; python3 resnet50.py ${OPTION}
cd ../../image_classification/vgg16; python3 vgg16.py ${OPTION}
cd ../../image_classification/squeezenet; python3 squeezenet.py ${OPTION}
cd ../../image_classification/efficientnet_lite; python3 efficientnet_lite.py ${OPTION}
cd ../../image_classification/googlenet; python3 googlenet.py ${OPTION}
cd ../../image_segmentation/deeplabv3plus; python3 deeplabv3plus.py ${OPTION}
cd ../../image_segmentation/hrnet_segmentation; python3 hrnet_segmentation.py ${OPTION}
#cd ../../image_segmentation/segment-anything-2; python3 segment-anything-2.py ${OPTION}
cd ../../object_detection/yolov3-tiny; python3 yolov3-tiny.py ${OPTION}
cd ../../object_detection/yolox; python3 yolox.py ${OPTION}
cd ../../object_detection/yolox; python3 yolox.py -m yolox_s ${OPTION}
cd ../../object_detection/efficientdet_lite; python3 efficientdet_lite.py ${OPTION}
cd ../../pose_estimation/pose_resnet; python3 pose_resnet.py ${OPTION}
cd ../../super_resolution/srresnet; python3 srresnet.py ${OPTION}
cd ../../super_resolution/espcn; python3 espcn.py ${OPTION}
