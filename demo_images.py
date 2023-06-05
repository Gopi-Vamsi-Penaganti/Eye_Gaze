import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def get_face(frame):
    # Load the pre-trained face cascade from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # Extract the face image from the frame
        face_image = frame[y:y+h, x:x+w]
        break 
    face_image = cv2.resize(face_image,(448,448))
    return face_image

def gaze_location(face, model):
    face_input = np.expand_dims(face, axis=0)
    location = model.predict(face_input)[0]
    return location

def load_saved_model(path='saved_models/best_model.h5'):
    return load_model(path, compile=False)

img = cv2.imread('Images/face_1.jpg')

window_name = 'Input Image'
cv2.imshow(window_name, img)


# Extract face from image
face_image = get_face(img)
# show face
window_name = 'Extracted Face'
cv2.imshow(window_name, face_image)

# Predict Gaze Location
model = load_saved_model()
gaze_loc = gaze_location(face_image, model)
print('PREDICTED GAZE LOCATION')
print(gaze_loc)
x = int(gaze_loc[0] * img.shape[1]/1000)
y = int(gaze_loc[1] * img.shape[0]/1000)
cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
cv2.imshow('Webcam', img)



cv2.waitKey(0)
cv2.destroyAllWindows()