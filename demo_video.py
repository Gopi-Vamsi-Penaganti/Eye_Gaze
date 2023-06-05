import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def get_face(frame):
    # Load the pre-trained face cascade from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces)>0:
        for (x, y, w, h) in faces:
            # Extract the face image from the frame
            face_image = frame[y:y+h, x:x+w]
            break 
        face_image = cv2.resize(face_image,(448,448))
        return face_image
    return None

def gaze_location(face, model):
    face_input = np.expand_dims(face, axis=0)
    location = model.predict(face_input)[0]
    return location

def load_saved_model(path='saved_models/best_model.h5'):
    return load_model(path, compile=False)

if __name__ == "__main__":
    # Load the saved model
    model = load_saved_model()

    cap = cv2.VideoCapture(0)  # Use the default webcam
    # Get the full screen resolution
    screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)  # Set the frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)  # Set the frame height
    frame_num = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_num+=1
        # Get the face from the frame
        face = get_face(frame)

        # If a face is detected
        if face is not None:
            # Get the gaze location
            gaze_loc = gaze_location(face, model)

            # Plot the gaze location on the frame
            x = int(gaze_loc[0] * frame.shape[1]/1000)
            y = int(gaze_loc[1] * frame.shape[0]/1000)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Display the resulting frame
        cv2.putText(frame, f"Frame: {frame_num}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
