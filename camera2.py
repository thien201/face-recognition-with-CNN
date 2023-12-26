import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("khuonmat_augmented.h5")

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a handle to the default webcam
cap = cv2.VideoCapture(0)

# Define the class names based on the label binarizer
class_names = ['Cavani', 'Mbappe', 'Messi', 'Obama', 'Ronaldo', 'thien']

# Start the webcam loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the region of interest (the face)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (100, 100))  # Resize to match the input size of the model
        roi_gray = roi_gray.reshape((1, 100, 100, 1))  # Reshape to model's expected input shape
        roi_gray = roi_gray / 255.0  # Normalize pixel values as done during training

        # Predict the class (person) of the face
        result = model.predict(roi_gray)
        final = np.argmax(result, axis=1)[0]  # Get the index of the max value
        player_name = class_names[final]
        probability = result[0][final]  # Get the probability of the predicted class

        # Put the class name and probability on the image
        cv2.putText(frame, f"{player_name} ({probability:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()