import cv2
import os
import apriltag

# Path to the directory containing the images
image_dir = 'images/apriltag'

# Iterate over each image in the directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)

    # Convert image to grayscale for AprilTag detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detector = apriltag.Detector()
    result = detector.detect(gray)

    # Create a mask of the same size as the image, initialized to white
    mask = 255 * np.ones_like(gray)

    for tag in result:
        # Get the coordinates of the AprilTag's corners
        corners = tag['lb-rb-rt-lt'].astype(int)
        # Fill the detected AprilTag region with black in the mask
        cv2.fillConvexPoly(mask, corners, 0)

    # Convert the white background to a slightly gray color (e.g., (240, 240, 240))
    image[mask == 255] = [240, 240, 240]

    # Save the processed image
    cv2.imwrite(os.path.join(image_dir, 'processed_' + image_name), image)
