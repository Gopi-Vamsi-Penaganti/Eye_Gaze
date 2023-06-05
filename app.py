import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import gdown
from streamlit_webrtc import webrtc_streamer

def get_face(frame):
    # Load the pre-trained face cascade from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_image = frame[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (448, 448))
        return face_image
    return None

def gaze_location(face, model):
    face_input = np.expand_dims(face, axis=0)
    location = model.predict(face_input)[0]
    return location

def load_saved_model(path='saved_models/best_model.h5'):
    return load_model(path, compile=False)

@st.cache
def download_model():
    file_id = '18ZDtFwPYxF9-4GbOOZzWIKltBkjVESFT'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'saved_models/best_model.h5'
    gdown.download(url, output, quiet=False)

def main():
    # Load the saved model
    download_model()
    model = load_saved_model()

    st.title("Face Gaze Detection using webcam")
    run = st.checkbox('Run')

    while run:
        webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=None, async_transform=True)

        if webrtc_ctx.video_transformer:
            frame = webrtc_ctx.video_transformer.input_frame
        else:
            st.warning("Webcam stream disconnected.")
            break

        # Get the face from the frame
        face = get_face(frame)

        # If a face is detected
        if face is not None:
            # Get the gaze location
            gaze_loc = gaze_location(face, model)

            # Plot the gaze location on the frame
            x = int(gaze_loc[0] * frame.shape[1] / 1000)
            y = int(gaze_loc[1] * frame.shape[0] / 1000)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Display the resulting frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB")

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
