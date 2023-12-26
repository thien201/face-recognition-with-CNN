import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
save_model = tf.keras.models.load_model("khuonmat_augmented.h5")

# Load the image file
filename = '16.jpg'
image = cv2.imread(filename)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Font for the text to be written on the image
fontface = cv2.FONT_HERSHEY_SIMPLEX

# Define the class names based on the label binarizer
# This should be in the same order as during training
class_names = ['Cavani', 'Mbappe', 'Messi', 'Obama', 'Ronaldo', 'thien']

# Iterate over each detected face
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract the region of interest (the face)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (100, 100))  # Resize to match the input size of the model
    roi_gray = roi_gray.reshape((1, 100, 100, 1))  # Reshape to model's expected input shape
    roi_gray = roi_gray / 255.0  # Normalize pixel values as done during training

    # Predict the class (player) of the face
    result = save_model.predict(roi_gray)
    final = np.argmax(result, axis=1)[0]  # Get the index of the max value
    probabilities = result[0][final]  # Get the probability of the predicted class

    # Put the class name on the image
    player_name = class_names[final]
    cv2.putText(image, player_name, (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)

    # Print the prediction to the terminal
    print(f"Detected {player_name} with probability {probabilities:.2f}")

# Display the image with the predictions
cv2.imshow('Recognition', image)

# Wait for a key press and then close all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()