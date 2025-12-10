set OPTION=-b
cd ..\\
cd face_detection\\blazeface& python blazeface.py %OPTION%
cd ..\\..\\face_recognition\\facemesh& python facemesh.py %OPTION%
cd ..\\..\\hand_recognition\\blazehand& python blazehand.py %OPTION%
cd ..\\..\\image_classification\\efficientnet_lite& python efficientnet_lite.py %OPTION%
cd ..\\..\\image_classification\\mobilenetv1& python mobilenetv1.py %OPTION%
cd ..\\..\\image_classification\\mobilenetv2& python mobilenetv2.py %OPTION%
cd ..\\..\\image_classification\\resnet50& python resnet50.py %OPTION%
cd ..\\..\\image_classification\\resnet50& python resnet50.py %OPTION% --float
cd ..\\..\\image_classification\\googlenet; python3 googlenet.py %OPTION%
cd ..\\..\\image_segmentation\\deeplabv3plus& python deeplabv3plus.py %OPTION%
cd ..\\..\\image_segmentation\\hrnet_segmentation& python3 hrnet_segmentation.py %OPTION%
cd ..\\..\\image_segmentation\\segment-anything-2& python3 segment-anything-2.py %OPTION%
cd ..\\..\\object_detection\\yolov3-tiny& python yolov3-tiny.py %OPTION%
cd ..\\..\\object_detection\\yolox& python yolox.py %OPTION%
cd ..\\..\\object_detection\\efficientdet_lite& python efficientdet_lite.py %OPTION%
cd ..\\..\\super_resolution\\srresnet; python3 srresnet.py %OPTION%
cd ..\\..\\super_resolution\\espcn; python3 espcn.py %OPTION%
