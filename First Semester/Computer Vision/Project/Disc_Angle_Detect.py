import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1_path = "/home/xgang/XGang/Graduation/First_Year/Computer Vision/Project/Figures/IMG_6852.jpg"
image2_path = "/home/xgang/XGang/Graduation/First_Year/Computer Vision/Project/Figures/IMG_6858.jpg"
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Convert images to RGB for consistent plotting
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Convert the second image to HSV for better color segmentation
image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

# Define the yellow color range for masking
lower_yellow = np.array([20, 100, 100])  # Adjusted for general yellow hue
upper_yellow = np.array([40, 255, 255])

# Create a mask for the yellow color
mask = cv2.inRange(image2_hsv, lower_yellow, upper_yellow)

# Apply morphological operations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# Find contours in the cleaned mask
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and select the largest one (assumed to be the disc)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
if contours:
    largest_contour = contours[0]

    # Fit an ellipse to the largest contour
    if len(largest_contour) >= 5:  # Minimum points required to fit an ellipse
        ellipse = cv2.fitEllipse(largest_contour)

        # Draw the ellipse on the original image for visualization
        image_with_ellipse = image2_rgb.copy()
        cv2.ellipse(image_with_ellipse, ellipse, (255, 0, 0), 2)

        # Extract ellipse parameters: center, axes, and angle
        center, axes, angle = ellipse
        major_axis, minor_axis = axes

        # Display results
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(mask_cleaned, cmap='gray')
        plt.title("Yellow Mask")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(image_with_ellipse)
        plt.title("Ellipse Fit")
        plt.axis("off")
        plt.show()

        ellipse_info = {
            "Center": center,
            "Major Axis": major_axis,
            "Minor Axis": minor_axis,
            "Angle": angle
        }
        ellipse_info
    else:
        "Not enough points to fit an ellipse."
else:
    "No contours detected."

# Prepare to draw the XYZ axes on the detected ellipse
# The center of the ellipse will be the origin (x=0, y=0, z=0 in the image plane)
center_x, center_y = map(int, ellipse[0])  # Ellipse center
major_axis_length = axes[0] / 2  # Half the major axis
minor_axis_length = axes[1] / 2  # Half the minor axis
angle_deg = angle  # Angle of ellipse rotation

# Draw the axes on the ellipse image
axes_image = image_with_ellipse.copy()

# Calculate the points on the ellipse's edge corresponding to the highest and lowest positions
# These are based on the center, major/minor axes, and angle

# Calculate the ellipse rotation in radians
angle_rad = np.deg2rad(angle_deg)

# Find the highest and lowest points on the ellipse edge
highest_point = (int(center_x - minor_axis_length * np.sin(angle_rad)),
                 int(center_y - minor_axis_length * np.cos(angle_rad)))
lowest_point = (int(center_x + minor_axis_length * np.sin(angle_rad)),
                int(center_y + minor_axis_length * np.cos(angle_rad)))

# Draw these points on the image for visualization
highlighted_image = axes_image.copy()
cv2.circle(highlighted_image, highest_point, 10, (0, 255, 255), -1)  # Yellow for highest point
cv2.circle(highlighted_image, lowest_point, 10, (255, 255, 0), -1)  # Cyan for lowest point

# Display the image with highlighted points
plt.figure(figsize=(8, 8))
plt.imshow(highlighted_image)
plt.title("Highest and Lowest Points on Disc Edge")
plt.axis("off")
plt.show()

# Return the coordinates of the highest and lowest points
highest_and_lowest = {
    "Highest Point": {"x": highest_point[0], "y": highest_point[1]},
    "Lowest Point": {"x": lowest_point[0], "y": lowest_point[1]}
}
print(highest_and_lowest)

