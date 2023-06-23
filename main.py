import cv2

# Load the Haar cascade XML file for full body detection
cascade_path = 'haarcascade_fullbody.xml'
fullbody_cascade = cv2.CascadeClassifier(cascade_path)

# Load the image
image_path = 'image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect full bodies in the image
fullbodies = fullbody_cascade.detectMultiScale(gray)

# Count the number of detected full bodies
num_people = len(fullbodies)

# Print the result
print(f"Number of people in the image: {num_people}")
