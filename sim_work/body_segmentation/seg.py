import numpy as np
import cv2

from body_mask import get_human_mask, get_body_mask

def get_key_points(mask):
    laplacian_line = cv2.Laplacian(mask, 2)
    laplacian_line = np.array(laplacian_line, np.uint8)
    
    cv2.imshow('lap', laplacian_line)
    cv2.waitKey(0)
    return

def human_distances():
    human_mask = get_human_mask(img_path = '1.png')
    body_mask = get_body_mask(human_mask)
    
    key_points = get_key_points(body_mask)
    
    print(key_points)
    
    return


def _main():
    human_distances()

if __name__ == "__main__":
    _main()