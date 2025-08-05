import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video path
video_path = '/home/xgang/XGang/Graduation/First_Year/Computer Vision/Assignment1/trainData/square/train_square3.mp4'  # Replace with your video file
output_path = 'mediapipe_left_hand_tracking.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# List to store the motion trail of the 17th point
motion_trail = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw the hand landmarks and track the 17th point (PINKY_MCP)
    if results.multi_hand_landmarks and results.multi_handedness:
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
                cv2.putText(frame, f"Z: {z:.2f}", (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw the motion trail
        for point in motion_trail:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
hands.close()

print("Processed video saved at:", output_path)
