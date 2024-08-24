import numpy as np
import cv2

from configs import K_SIZE

def get_key_points(mask):
    gray = np.float32(mask)
    dst = cv2.cornerHarris(gray,2,3,0.05)
    
    cv2.imshow('lan', dst)
    cv2.waitKey(0)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    flags = np.array(dst > 100)
    cand_points = []
    for i, row in enumerate(flags):
        row = list(row)
        if True not in row: continue
        cand_points.append([i, row.index(True)])
            
    cands = []
    tmp_group = []
    for cand_point in cand_points:
        
        if tmp_group == []: 
            tmp_group.append(cand_point)
            continue
    
        dis_group = (tmp_group[-1][0] - cand_point[0]) ** 2 + (tmp_group[-1][1] - cand_point[1]) ** 2 
        if dis_group > 49:
            cands.append(tmp_group[len(tmp_group) // 2])
            tmp_group = []

    (h, w, _) = mask.shape
    
    real_cands = []
    for cand_point in cands:
        y1 = cand_point[1] - K_SIZE if cand_point[1] > K_SIZE else 0
        y2 = cand_point[1] + K_SIZE if cand_point[1] + K_SIZE < h else h
        x1 = cand_point[0] - K_SIZE if cand_point[0] > K_SIZE else 0
        x2 = cand_point[0] + K_SIZE if cand_point[0] + K_SIZE < w else w
        
        window = mask[x1: x2, y1: y2]
        if np.sum(window) / (255 * (K_SIZE * 2 + 1) ** 2) > 2:
            real_cands.append(cand_point)

    return real_cands