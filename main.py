import cv2
import numpy as np
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_pose = mp.solutions.pose


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Recolor image to RGB
        frame_ = cv2.imread('9.png')
        image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
        width = int(frame_.shape[1])
        height = int(frame_.shape[0])

        image.flags.writeable = False
    
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            head = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            left_diff_elbow_sho = calculate_distance(left_shoulder, left_elbow)*height  /3
            left_diff_waist_hip = calculate_distance(left_shoulder, left_hip) * height /3
            # right_diff_elbow_sho = calculate_distance(right_shoulder, right_elbow)*height  /3

            # wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            # hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            # right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            # left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            # right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            # left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            
            # right_knee_3d = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z])
            # left_knee_3d = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z])
            # right_ank_3d = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z])
            # left_ank_3d = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z])
            # right_hip_3d = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])
            # left_hip_3d = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z])

            # xx = right_knee
            cv2.circle(image, (int(right_shoulder[0]*width), int(right_shoulder[1]*height) + int(left_diff_elbow_sho)), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(left_shoulder[0]*width), int(left_shoulder[1]*height) + int(left_diff_elbow_sho)), 3, (0, 0, 255), -1)

            cv2.circle(image, (int(left_shoulder[0]*width), int(left_hip[1]*height) - int(left_diff_waist_hip)), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(right_shoulder[0]*width), int(left_hip[1]*height) - int(left_diff_waist_hip)), 3, (0, 0, 255), -1)

            

            left_chest = [int(right_shoulder[0]*width), int(right_shoulder[1]*height) + int(left_diff_elbow_sho)]
            right_chest = [int(left_shoulder[0]*width), int(left_shoulder[1]*height) + int(left_diff_elbow_sho)]

            left_waist = [int(right_shoulder[0]*width),  int(left_hip[1]*height - left_diff_waist_hip )]
            right_waist = [int(left_shoulder[0]*width),  int(left_hip[1]*height - left_diff_waist_hip )]
            cv2.line(image, (left_chest[0], left_chest[1]), (right_chest[0], right_chest[1]),(0, 255, 255) , 3)
            cv2.line(image, (left_waist[0], left_waist[1]), (right_waist[0], right_waist[1]),(0, 255, 255) , 3)
            # cv2.line(image, (100, 100), (100, 100), (0, 255, 255), 3)
            # cv2.circle(image, (int(right_shoulder[0]*width), 10), 10, (0, 0, 255), -1)
            # cv2.circle(image, landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER], 10, (0, 0, 255), -1)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(203,17,17), thickness=2, circle_radius=2),
                                     ) 
            
            # cv2.circle(image, (right_knee[0]*width, right_knee[1]*height), 3, (0, 255, 0), 2) 
            cv2.imshow('result.png', image)
            cv2.waitKey(0)
        except:
            pass
        # pos_x = [left_ankle[0]*width, right_ankle[0]*width, left_knee[0]*width, right_knee[0]*width, left_shoulder[0]*width, right_shoulder[0]*width, head[0]*width]
        # pos_y = [left_ankle[1]*height, right_ankle[1]*height, left_knee[1]*height, right_knee[1]*height, left_shoulder[1]*height, right_shoulder[1]*height, head[1]*height]


