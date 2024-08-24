from flask import *
import cv2
import threading
import cv2
import numpy as np
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from ruamel import yaml
from ultralytics import YOLO
from utils import detect
from utils import draw_results
from openvino.runtime import Core
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



# model = YOLO("yolov8n-pose")

data = yaml.safe_load(open('yolov8n/metadata.yaml'))
seg_model_path = 'yolov8n/yolov8n-seg.xml'
label_map = data['names']

core = Core()
seg_ov_model = core.read_model(seg_model_path)
device = "CPU"  # GPU
if device != "CPU":
    seg_ov_model.reshape({0: [1, 3, 640, 640]})
seg_compiled_model = core.compile_model(seg_ov_model, device)

image  = cv2.imread('1.png')


detections = detect(image, seg_compiled_model)[0]
# if len(detections['det']) == 0:
#     continue
dets = detections['det']
segs = detections['segment']


draw_results(image, segs, label_map)

cv2.waitKey(0)








