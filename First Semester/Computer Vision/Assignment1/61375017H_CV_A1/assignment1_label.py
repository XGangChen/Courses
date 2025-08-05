import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
from glob import glob

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Paths for training and testing datasets
train_data = {
    'Square': glob('/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/trainData/square/*.mp4'),
    'Triangle': glob('/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/trainData/triangle/*.mp4'),
    'Wave': glob('/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/trainData/wave/*.mp4'),
    'Star': glob('/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/trainData/star/*.mp4')
}
test_data = {
    'Square': glob('/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/testData/square/*.mp4'),
    'Triangle': glob('/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/testData/triangle/*.mp4'),
    'Wave': glob('/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/testData/wave/*.mp4'),
    'Star': glob('/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/testData/star/*.mp4')
}

# Classification and evaluation
def approximate_and_classify(trail, epsilon_factor = 0.02):
    if len(trail) < 5:
        return "Undetected"

    points = np.array(trail, dtype=np.int32)
    hull = cv2.convexHull(points)
    epsilon = epsilon_factor * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    num_vertices = len(approx)

    # Calculate curvature (change in angle) for the trail
    def calculate_curvature(trail):
        angles = []
        for i in range(1, len(trail) - 1):
            a = np.array(trail[i - 1])
            b = np.array(trail[i])
            c = np.array(trail[i + 1])
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(np.degrees(angle))
        return angles
    
    angles = calculate_curvature(trail)
    sharp_turns = sum(1 for angle in angles if angle < 60)  # Count sharp angles 

    
    # Classification rules
    if num_vertices == 3:
        return "Triangle"
    elif calculate_repetitions(trail) > 2:
        return "Wave"
    elif 4 <= num_vertices <= 5:
        return "Square"
    # elif num_vertices > 10:
    #     return "Wave"
    # elif 7 <= num_vertices <= 9:
    #     return "Star"
    elif num_vertices >= 6 or sharp_turns > 2:  # Many sharp turns for stars
        return "Star"
    elif num_vertices > 10 or len(trail) > 30:  # Longer trails for waves
        return "Wave"
    # if num_vertices > 10 or len(trail) > 30:  # Longer trails for waves
    #     return "Wave"
    # elif num_vertices >= 6 or sharp_turns > 2:  # Many sharp turns for stars
    #     return "Star"
    # elif 4 <= num_vertices <= 5:
    #     return "Square"
    # elif num_vertices == 3:
    #     return "Triangle"
    return "Undetected"

def calculate_repetitions(trail):
    y_coords = [point[1] for point in trail]
    peaks = sum(1 for i in range(1, len(y_coords) - 1) if y_coords[i - 1] < y_coords[i] > y_coords[i + 1])
    return peaks

# Smooth the motion trail
def smooth_trail(trail, window_size=50):
    if len(trail) < window_size:
        return trail
    
    smoothed_trail = []
    for i in range(len(trail) - window_size + 1):
        window = trail[i:i + window_size]
        avg_x = int(np.mean([p[0] for p in window]))
        avg_y = int(np.mean([p[1] for p in window]))
        smoothed_trail.append((avg_x, avg_y))
    return smoothed_trail

# Process a video and return the detected shape
def process_video(video_path, label=None, epsilon_factor=0.02):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    motion_trail = []
    detected_shapes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            x = int(pinky_mcp.x * frame_width)
            y = int(pinky_mcp.y * frame_height)

            motion_trail.append((x, y))
            motion_trail = smooth_trail(motion_trail)

            detected_shape = approximate_and_classify(motion_trail, epsilon_factor)
            if detected_shape != "Undetected":
                detected_shapes.append(detected_shape)

            if len(motion_trail) > 100:
                motion_trail.pop(0)

    cap.release()

    if detected_shapes:
        final_result = Counter(detected_shapes).most_common(1)[0][0]
    else:
        final_result = "Undetected"

    return final_result, final_result == label if label else None

# Train the model
def train_model(train_data):
    print("Training:")
    for label, videos in train_data.items():
        for video_path in videos:
            detected_shape, _ = process_video(video_path, label)
            print(f"Video: {video_path} | Label: {label} | Detected: {detected_shape}")

# Test the model and evaluate accuracy
def test_model(test_data):
    print("\nTesting:")
    correct_predictions = 0
    total_tests = 0

    for label, videos in test_data.items():
        for video_path in videos:
            total_tests += 1
            detected_shape, is_correct = process_video(video_path, label)
            print(f"Video: {video_path} | Label: {label} | Detected: {detected_shape}")
            if is_correct:
                correct_predictions += 1

    accuracy = correct_predictions / total_tests
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Train and test the model
train_model(train_data)
test_model(test_data)

hands.close()
