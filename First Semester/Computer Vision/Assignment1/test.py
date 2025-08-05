import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video path and output settings
video_path = '/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/2024-11-25 08-50-48.mp4'
output_path = 'Sample_Video2.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# List to store motion trails and other variables
motion_trail = []

# List to store detected shapes
detected_shapes = []


def approximate_and_classify(trail, epsilon_factor):
    """
    Approximates the motion trail into lines and classifies the symbol based on angles.

    Parameters:
        trail (list of tuple): Motion trail points.
        epsilon_factor (float): Sensitivity for approximation (lower = more precise).

    Returns:
        str: The detected symbol ("Square", "Triangle", "Star", "Wave", or "Undetected").
    """
    if len(trail) < 5:  # Need at least 5 points to approximate
        return "Undetected"

    # Convert trail to numpy array
    points = np.array(trail, dtype=np.int32)

    # Fit a convex hull to the points
    hull = cv2.convexHull(points)

    # Approximate the hull into a polygon
    epsilon = epsilon_factor * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Count the number of vertices in the approximated polygon
    num_vertices = len(approx)

    # Classify based on the number of vertices
    if num_vertices == 2:
        return "Triangle"
    elif num_vertices == 3:
        return "Square"
    elif num_vertices == 4:
        return "Star"
    elif num_vertices > 10:
        return "Wave"
    elif num_vertices > 50:
        return "Undetected"

# Helper function to detect shapes
# def detect_shape(trail):

#     if len(trail) < 10:
#         return "Undetected"

#     # Convert trail to numpy array
#     points = np.array(trail, dtype=np.int32)

#     # Fit a convex hull to the points
#     hull = cv2.convexHull(points)

#     # Approximate the hull into a polygon
#     epsilon = 0.02 * cv2.arcLength(hull, True)
#     approx = cv2.approxPolyDP(hull, epsilon, True)

#     # Detect shapes based on the number of vertices
#     if len(approx) == 2:
#         return "Triangle"
#     elif len(approx) == 3:
#         return "Square"
#     elif len(approx) > 5:
#         return "Wave"
#     elif len(approx) == 4:
#         return "Star"
#     elif len(approx) > 20:
#         return "Undetected"

# def add_to_trail(point, trail, max_distance=20):
#     if len(trail) == 0:
#         trail.append(point)
#     else:
#         # Calculate the distance from the last point
#         last_point = trail[-1]
#         distance = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
        
#         # Only add the point if the distance is within the threshold
#         if distance < max_distance:
#             trail.append(point)

def smooth_trail(trail, window_size=50):
    if len(trail) < window_size:
        return trail
    
    smoothed_trail = []
    for i in range(len(trail) - window_size + 1):
        # Compute the average of the window
        window = trail[i:i + window_size]
        avg_x = int(np.mean([p[0] for p in window]))
        avg_y = int(np.mean([p[1] for p in window]))
        smoothed_trail.append((avg_x, avg_y))
    
    return smoothed_trail


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        furthest_hand_landmark = None
        max_distance = float('-inf')

        # Determine the furthest hand
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            z_mean = np.mean([lm.z for lm in hand_landmarks.landmark])
            if z_mean > max_distance:
                max_distance = z_mean
                furthest_hand_landmark = hand_landmarks

        # Process the furthest hand
        if furthest_hand_landmark:
            # Extract the 17th landmark (PINKY_MCP)
            pinky_mcp = furthest_hand_landmark.landmark[mp_hands.HandLandmark.PINKY_MCP]
            x = int(pinky_mcp.x * frame_width)
            y = int(pinky_mcp.y * frame_height)

            # Add the point to the motion trail
            motion_trail.append((x, y))

            # add_to_trail((x, y), motion_trail)

            motion_trail = smooth_trail(motion_trail)



            # Draw the trail and classify the shape
            for point in motion_trail:
                cv2.circle(frame, point, 3, (0, 0, 255), -1)

            # Detect and classify the symbol
            detected_shape = approximate_and_classify(motion_trail, epsilon_factor=0.02)

            # Add the detected shape to the list if it's not "Undetected"
            if detected_shape != "Undetected":
                detected_shapes.append(detected_shape)

            # Display the detected shape on the video frame
            cv2.putText(frame, f"Drawing: {detected_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Limit the trail length
            if len(motion_trail) > 100:
                motion_trail.pop(0)

    # Write the processed frame to the output video
    out.write(frame)


# Calculate the most common shape
if detected_shapes:
    final_result = Counter(detected_shapes).most_common(1)[0][0]
else:
    final_result = "Undetected"

print("Final result: The symbol most likely drawn is:", final_result)


# Release resources
cap.release()
out.release()
hands.close()

output_path
