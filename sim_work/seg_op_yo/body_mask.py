from ultralytics import YOLO
import numpy as np
import cv2

def get_human_mask(img_path):
    model = YOLO('seg_models/yolov8n-seg.pt')  # load an official model

    results = model(img_path)[0]  # predict on an image
    image = cv2.imread(img_path)

    (h, w, _) = image.shape
    black_image = np.zeros((h, w), np.uint8)

    masks = results.masks

    area = 0
    for item in masks:
        mask = item.data[0].numpy()
        
        if np.sum(mask) > area:
            area = np.sum(mask)
            human_mask = mask

    return human_mask

def get_body_mask(human_mask):
    human_mask = np.array(human_mask, np.uint8) * 255
    contours, _ = cv2.findContours(human_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    (hh, ww) = human_mask.shape
    [x, y, w, h] = cv2.boundingRect(contours[0])
    cropped_mask = human_mask[y - 5 if y > 5 else 0: h + y + 5 if h + y + 5 < hh else hh, x - 5 if x > 5 else 0: w + x + 5 if w + x + 5 < ww else ww]
    # cropped_mask = human_mask[y: y + h, x: x + w]
    cv2.imshow('human_body', human_mask)
    cv2.imshow('cropped', cropped_mask)
    cv2.waitKey(0)
    
    return cropped_mask, [x - 5 if x > 5 else 0, y - 5 if y > 5 else 0, w, h]