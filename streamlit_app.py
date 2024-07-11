import streamlit as st
import joblib
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tempfile

# Load your decision tree model
decision_tree_model = joblib.load('decision_tree_model.joblib')

# Load your CNN model
cnn_model = load_model('cnn_model.h5')

def predict_decision_tree(input_data):
    prediction = decision_tree_model.predict([input_data])[0]
    return prediction

def preprocess_video(video_path, max_frames=50, resize_shape=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_shape)
        frames.append(frame)
    cap.release()
    if len(frames) < max_frames:
        for _ in range(max_frames - len(frames)):
            frames.append(np.zeros(resize_shape + (3,)))
    return np.array(frames)

def predict_video(model, video_path):
    video_frames = preprocess_video(video_path)
    video_frames = np.expand_dims(video_frames, axis=0)
    prediction = model.predict(video_frames)
    return np.argmax(prediction, axis=1)[0]

# Main content area
st.title("AI Model Deployment")

st.header("Decision Tree Model Input")
st.write("Input the sensor data for the decision tree model")

accx = st.number_input('AccX')
accy = st.number_input('AccY')
accz = st.number_input('AccZ')
gyrox = st.number_input('GyroX')
gyroy = st.number_input('GyroY')
gyroz = st.number_input('GyroZ')
timestamp = st.number_input('Timestamp')

input_data = [accx, accy, accz, gyrox, gyroy, gyroz, timestamp]

if st.button('Predict Decision Tree'):
    decision_tree_prediction = predict_decision_tree(input_data)
    decision_tree_labels = ["", "Aggressive", "Normal", "Slow"]
    st.write(f"Decision Tree Prediction: {decision_tree_labels[int(decision_tree_prediction)]}")

st.header("CNN Model Input")
st.write("Upload a video file for the CNN model")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

if st.button('Predict CNN'):
    cnn_prediction = predict_video(cnn_model, video_path)
    result = "Rash" if cnn_prediction == 0 else "Smooth"
    st.write(f"CNN Prediction: {result}")
