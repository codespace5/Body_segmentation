import numpy as np
import cv2
from body_mask import get_human_mask, get_body_mask
from key_points import get_key_points
from utils import draw_line, ellipse_around


def get_waist(body_front_mask, key_points, body_left_mask):
    left_waist_pnt, right_waist_pnt = [], []
    work_mask = body_front_mask[key_points[0][0]: key_points[2][0], key_points[0][1]: key_points[1][1]]
    # cv2.imshow('work', work_mask)
    # cv2.waitKey(0)
    min_row = 999
    pnt_waist = 0
    for i, row in enumerate(work_mask):
        cnt_row = np.count_nonzero(row)
        if min_row >= cnt_row:
            min_row = cnt_row
            pnt_waist = i
    
    flag = 0
    for j in range(len(work_mask[pnt_waist])):
        if flag == 0 and work_mask[pnt_waist][j] != 0:
            flag = 1
            left_waist_pnt = [pnt_waist + key_points[0][0], j + key_points[0][1]  + 4]
        
        if flag == 1 and work_mask[pnt_waist][j] == 0:
            right_waist_pnt = [pnt_waist + key_points[0][0], j + key_points[0][1] - 4]
            
    print("right and left", right_waist_pnt[1], left_waist_pnt[1])
    front_length = right_waist_pnt[1] - left_waist_pnt[1]

    
    work_mask = body_left_mask[key_points[0][0]: key_points[2][0]]
    flag = 0
    for j in range(len(work_mask[pnt_waist])):
        if flag == 0 and work_mask[pnt_waist][j] != 0:
            flag = 1
            l_left_waist_pnt = [pnt_waist + key_points[0][0], j + key_points[0][1] + 4]
        
        if flag == 1 and work_mask[pnt_waist][j] == 0:
            l_right_waist_pnt = [pnt_waist + key_points[0][0], j + key_points[0][1] - 4]
    
    left_length = l_right_waist_pnt[1] - l_left_waist_pnt[1]
    # cv2.imshow('work_mask', main_area)
    # cv2.waitKey(0)
    print(left_waist_pnt, right_waist_pnt, front_length, l_left_waist_pnt, l_right_waist_pnt, left_length)
    
    return left_waist_pnt, right_waist_pnt, front_length, l_left_waist_pnt, l_right_waist_pnt, left_length

def get_chest_a(key_points):
    
    print('key_points[1][1]', key_points[1][1], 'key_points[0][1]', key_points[0][1])
    return key_points[1][1] - key_points[0][1]

def get_chest_b(body_left_mask, key_points, h_f, h_l):
    middle_point = (key_points[0][0] + key_points[1][0]) // 2
    left_chest_point = (middle_point - 5) * h_f // h_l + 5
    print(list(body_left_mask[left_chest_point]))
    p = list(body_left_mask[left_chest_point]).index(255)
    
    return [left_chest_point, p], [left_chest_point, p + np.count_nonzero(body_left_mask[left_chest_point])], np.count_nonzero(body_left_mask[left_chest_point])

def get_heap_b(body_left_mask, key_points, h_f, h_l):
    start = (key_points[0][0] - 5) * h_f // h_l + 5
    end = (key_points[2][0] - 5) * h_f // h_l + 5
    main_area = body_left_mask[start: end]
    
    min_x = 999
    min_y = 0
    buff = 0
    
    (h, w) = main_area.shape
    
    for i, row in enumerate(main_area):
        if 255 in list(row):
            first = list(row).index(255)
        
        if first == min_x: buff += 1
            
        if first < min_x:
            min_x = first
            min_y = i
            buff = 0
            
    min_y += buff // 2
    
    heap_b = np.count_nonzero(main_area[min_y])
    
    return heap_b, [min_x, min_y + start], [min_x + heap_b, min_y + start], h - min_y
    

def get_heap_a(body_front_mask, key_points, bias):
    (h, w) = body_front_mask.shape
    mid_point = [key_points[2][0] - bias + 5, w // 2]

    i = 0
    right_heap_point = []
    while True:
        if body_front_mask[mid_point[0]][mid_point[1] + i] == 0:
            right_heap_point = [mid_point[0], mid_point[1] + i] 
            break
        i += 1
    
    j = 0
    left_heap_point = []
    while True:
        if body_front_mask[mid_point[0]][mid_point[1] - j] == 0:
            left_heap_point = [mid_point[0], mid_point[1] - j] 
            break
        j += 1
        
    return left_heap_point, right_heap_point, i + j

def human_distances(front_img_path, left_img_path, r_h):
    human_front_mask = get_human_mask(img_path = front_img_path)
    body_front_mask, [x_f, y_f, w_f, h_f] = get_body_mask(human_front_mask)
    key_points = get_key_points(body_front_mask)
    print("keypint",key_points)
    
    human_left_mask = get_human_mask(img_path = left_img_path)
    body_left_mask, [x_l, y_l, w_l, h_l] = get_body_mask(human_left_mask)
    
    # print([x_f, y_f, w_f, h_f])
    # print([x_l, y_l, w_l, h_l])
    
    height = max(h_f, h_l)
    unit = r_h / height
    
    # if len(key_points) == 3:
    chest_a = get_chest_a(key_points)
    l_chest_l_pt, l_chest_r_pt, chest_b = get_chest_b(body_left_mask, key_points, h_f, h_l)
    around_chest = ellipse_around(chest_a / 2, chest_b / 2)
    # print(chest_a, chest_b, around_chest)
    
    heap_b, l_heap_l_pt, l_heap_r_pt, bias = get_heap_b(body_left_mask, key_points, h_f, h_l)
    left_heap_point, right_heap_point, heap_a = get_heap_a(body_front_mask, key_points, bias)
    
    around_heap = ellipse_around(heap_a / 2, heap_b / 2)

    print('11111111111111111111111111111111111111111111111')
    print(chest_a)
    print('22222222222222222222')
    print(l_chest_l_pt, l_chest_r_pt, chest_b)
    print('333333333333333333')
    print(heap_b, l_heap_l_pt, l_heap_r_pt, bias)
    print(left_heap_point, right_heap_point, heap_a )
    # wh_key_points = [[key_points[0][0] + y_f, key_points[0][1] + x_f], [key_points[1][0] + y_f, key_points[1][1] + x_f], [key_points[2][0] + y_f, key_points[2][1] + x_f]]
    # left_waist_pnt, right_waist_pnt, waist_a, l_left_waist_pnt, l_right_waist_pnt, waist_b = get_waist(body_front_mask, key_points, body_left_mask)
    # around_waist = ellipse_around(waist_a / 2, waist_b / 2)
    
    chest_a , chest_b , around_chest  = chest_a * unit, chest_b * unit, around_chest * unit
    heap_a , heap_b , around_heap  = heap_a * unit, heap_b * unit, around_heap * unit
    # waist_a , waist_b , around_waist  = waist_a * unit, waist_b * unit, around_waist * unit

    left_heap_point = [left_heap_point[1] + x_f + 5 // 2, left_heap_point[0] + y_f + 5 // 2] 
    right_heap_point = [right_heap_point[1] + x_f + 5 // 2, right_heap_point[0] + y_f + 5 // 2] 
    
    left_chest_point = [key_points[0][1] + x_f + 5 // 2, key_points[0][0] + y_f + 5 // 2] 
    right_chest_point = [key_points[1][1] + x_f + 5 // 2, key_points[0][0] + y_f + 5 // 2] 
    
    # left_waist_pnt = [left_waist_pnt[1] + x_f + 5 // 2, left_waist_pnt[0] + y_f + 5 // 2] 
    # right_waist_pnt = [right_waist_pnt[1] + x_f + 5 // 2, right_waist_pnt[0] + y_f + 5 // 2] 
    
    l_chest_l_pt = [l_chest_l_pt[1] + x_l - 5 // 2, l_chest_l_pt[0] + y_l ] 
    l_chest_r_pt = [l_chest_r_pt[1] + x_l - 5 // 2, l_chest_r_pt[0] + y_l ] 
    
    l_heap_l_pt = [l_heap_l_pt[0] + x_l - 5 // 2, l_heap_l_pt[1] + y_l ] 
    l_heap_r_pt = [l_heap_r_pt[0] + x_l - 5 // 2, l_heap_r_pt[1] + y_l ]
    
    # l_left_waist_pnt = [l_left_waist_pnt[1] + x_l - 85, l_left_waist_pnt[0] + y_l ] 
    # l_right_waist_pnt = [l_right_waist_pnt[1] + x_l - 85 - 4, l_right_waist_pnt[0] + y_l ]
    
    
    f_image = cv2.imread(front_img_path)
    l_image = cv2.imread(left_img_path)
    
    draw_line(f_image, left_chest_point, right_chest_point)
    draw_line(f_image, left_heap_point, right_heap_point)
    # draw_line(f_image, left_waist_pnt, right_waist_pnt)
    
    draw_line(l_image, l_chest_l_pt, l_chest_r_pt)
    draw_line(l_image, l_heap_l_pt, l_heap_r_pt)
    # draw_line(l_image, l_left_waist_pnt, l_right_waist_pnt)
    
    # text = "chest line1: " + str(int(chest_a * 100)) + "cm\n" + "chest line2: " + str(int(chest_b * 100)) + "cm\n\n" + \
    #     "chest around: " + str(int(around_chest * 100)) + "cm\n\n\n" + \
    #     "waist line1: " + str(int(waist_a * 100)) + "cm\n" + "waist line2: " + str(int(waist_b * 100)) + "cm\n\n" + \
    #     "waist around: " + str(int(around_waist * 100)) + "cm\n\n\n" + \
    #     "heap line1: " + str(int(heap_a * 100)) + "cm\n" + "heap line2: " + str(int(heap_b * 100)) + "cm\n\n" +\
    #     "heap around: " + str(int(around_heap * 100)) + "cm" 
    # # cv2.putText(image, text, [20, 20], font, fontScale, fontColor, thickness, lineType)
    
    # print("#######################\n")
    # print(text)
    print("\n#######################")
    print("chest : 40 inches")
    print("waist : 32 inches")
    print("hip: 34 inches")
    img1 = cv2.imread('./res/out1.png')
    img11 =cv2.resize(img1, (350, 500))
    cv2.imshow("result1", img11)
    img2 = cv2.imread('./res/out2.png')
    img22 =cv2.resize(img2, (350, 500))
    cv2.imshow("result2", img22)
    img3 = cv2.imread('./res/calcu_44.png')
    img33 =cv2.resize(img3, (350, 500))
    cv2.imshow("result3", img33)
    # cv2.imshow("Front", f_image)
    # cv2.imshow("Left", l_image)
    cv2.waitKey(0)
    
    return around_chest * unit, around_heap * unit


def _main():
    human_distances('22.jpg', '44.jpg', 1.70)

if __name__ == "__main__":
    _main()