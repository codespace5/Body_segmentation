import cv2
import mediapipe as mp
import math

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to calculate body measurements
def calculate_body_measurements(landmarks):
    # Check if required landmarks are present
    if mp.solutions.holistic.PoseLandmark.LEFT_HIP in landmarks and mp.solutions.holistic.PoseLandmark.RIGHT_HIP in landmarks:
        hip_width = calculate_distance(landmarks[mp.solutions.holistic.PoseLandmark.LEFT_HIP],
                                       landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_HIP])
    else:
        hip_width = 0  # or any default value

    if mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER in landmarks and mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER in landmarks:
        chest_width = calculate_distance(landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER],
                                         landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER])
    else:
        chest_width = 0  # or any default value

    # Calculate a virtual waist point as the midpoint between the shoulders
    if mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER in landmarks and mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER in landmarks:
        waist_point = (
            (landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER][0] + landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER][0]) // 2,
            (landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER][1] + landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER][1]) // 2
        )
    else:
        waist_point = (0, 0)

    # Calculate waist width as the distance between the virtual waist point and the hip center
    if mp.solutions.holistic.PoseLandmark.LEFT_HIP in landmarks and mp.solutions.holistic.PoseLandmark.RIGHT_HIP in landmarks:
        waist_width = calculate_distance(waist_point,
                                         ((landmarks[mp.solutions.holistic.PoseLandmark.LEFT_HIP][0] + landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_HIP][0]) // 2,
                                          (landmarks[mp.solutions.holistic.PoseLandmark.LEFT_HIP][1] + landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_HIP][1]) // 2))
    else:
        waist_width = 0  # or any default value

    return hip_width, chest_width, waist_width, waist_point


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()


frame = cv2.imread('1.png')


rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the frame with MediaPipe Holistic
results = holistic.process(rgb_frame)

if results.pose_landmarks:
    # Extract landmarks for measurements
    landmarks = {}
    for landmark in mp_holistic.PoseLandmark:
        landmarks[landmark] = (int(results.pose_landmarks.landmark[landmark].x * frame.shape[1]),
                                int(results.pose_landmarks.landmark[landmark].y * frame.shape[0]))

    # Calculate body measurements
    hip_width, chest_width, waist_width, waist_point = calculate_body_measurements(landmarks)

    # Draw lines to visualize the measurements on the frame
    cv2.line(frame, landmarks[mp.solutions.holistic.PoseLandmark.LEFT_HIP],
                landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_HIP], (0, 255, 0), 2)  # Hip width
    cv2.line(frame, landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER],
                landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER], (0, 0, 255), 2)  # Chest width
    cv2.line(frame, waist_point, ((landmarks[mp.solutions.holistic.PoseLandmark.LEFT_HIP][0] + landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_HIP][0]) // 2,
                                    (landmarks[mp.solutions.holistic.PoseLandmark.LEFT_HIP][1] + landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_HIP][1]) // 2), (255, 0, 0), 2)  # Waist width

    # Display measurements
    cv2.putText(frame, f'Hip Width: {hip_width:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Chest Width: {chest_width:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Waist Width: {waist_width:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Display the frame
cv2.imshow('Body Measurements', frame)



cv2.waitKey(0)