import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Predefined templates (displacement trajectories for each symbol)
templates = {
    "triangle": [(0.4, -0.8), (0.4, 0.8), (-0.8, 0)],  # Approximate relative displacements
    "square": [(0, 0.8), (0.8, 0), (0, -0.8), (-0.8, 0), (0, 0.8)],
    "wave": [(0.2, 0.2), (0.2, -0.4), (0.2, 0.4), (0.2, -0.4), (0.2, 0.2)],
    "star": [(0.1, -0.4), (0.2, 0.4), (0.3, 0), (-0.3, 0.4), (0.1, -0.4)],
}

# Video path
video_path = '/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/2024-11-13 06-52-07.mp4'  # Replace with your video file
output_path = 'Sample_Video.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Variables for trajectory and timer
displacements = []
motion_trail = []
symbol_detected = None
frames_per_interval = fps * 5  # Analyze every 5 seconds
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # # Process the frame with MediaPipe Hands
    # results = hands.process(rgb_frame)

    # # Draw the hand landmarks and track the 17th point (PINKY_MCP)
    # if results.multi_hand_landmarks and results.multi_handedness:
    #     # Iterate over the hands and handedness
    #     for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
    #         # Check if the hand is the right hand
    #         hand_label = handedness.classification[0].label
    #         if hand_label == 'Right':
    #             # Calculate the average x-coordinate of the hand landmarks to determine position
    #             hand_x_positions = [landmark.x for landmark in hand_landmarks.landmark]
    #             avg_hand_x = np.mean(hand_x_positions)

    #             # Check if the hand is on the left side of the frame
    #             if avg_hand_x < 0.5:
    #                 # Extract the 17th landmark (PINKY_MCP)
    #                 pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    #                 x = int(pinky_mcp.x * frame_width)
    #                 y = int(pinky_mcp.y * frame_height)
    #                 motion_trail.append((x, y))

    #                 # Calculate displacements
    #                 if len(motion_trail) > 1:
    #                     dx = (motion_trail[-1][0] - motion_trail[-2][0]) / frame_width
    #                     dy = (motion_trail[-1][1] - motion_trail[-2][1]) / frame_height
    #                     displacements.append((dx, dy))

    #                 # Draw the point on the frame
    #                 cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    #                 # Draw the motion trail
    #                 for point in motion_trail:
    #                     cv2.circle(frame, point, 3, (0, 0, 255), -1)

    #                 # Since we found the right hand on the left side, we can break out of the loop
    #                 break  # Remove this if you want to process all right hands on the left side

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw the hand landmarks and track the farthest hand
    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_label = handedness.classification[0].label  # "Left" or "Right"

            # Process only the left hand
            if hand_label == "Left":
                # Extract the 17th landmark (PINKY_MCP)
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                x = int(pinky_mcp.x * frame_width)
                y = int(pinky_mcp.y * frame_height)
                z = pinky_mcp.z  # Depth value

                # Add the point to the motion trail
                motion_trail.append((x, y))

                # Draw the point on the frame
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                # Annotate the z-value (optional)
                # cv2.putText(frame, f"Z: {z:.2f}", (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    # Analyze trajectory every 5 seconds
    frame_count += 1
    if frame_count >= frames_per_interval:
        if len(displacements) > 5:  # Ensure enough displacements for analysis
            # Match the displacement trajectory to templates
            best_match = None
            best_distance = float('inf')
            for symbol, template in templates.items():
                distance, _ = fastdtw(displacements, template, dist=euclidean)
                if distance < best_distance:
                    best_distance = distance
                    best_match = symbol

            symbol_detected = best_match

        # Reset for the next interval
        displacements = []
        motion_trail = []
        frame_count = 0

    # Display the recognized symbol
    if symbol_detected:
        cv2.putText(frame, f"Drawing: {symbol_detected}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
hands.close()

print("Processed video saved at:", output_path)
