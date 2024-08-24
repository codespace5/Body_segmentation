import numpy as np
import cv2

def draw_line(image, pt1, pt2):
    cv2.line(image, pt1, pt2, (255, 255, 0), 2)

def ellipse_around(a, b):
    h = (a - b) ** 2 / (a + b) ** 2
    
    return np.pi * (a + b) * (1 + (3 * h / (10 + np.sqrt(4 - 3 * h))))
