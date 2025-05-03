# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os, traceback
from PIL import Image
from moviepy.editor import VideoFileClip

# Keras imports
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image

app = Flask(__name__)
CORS(app)

# Load ML model and feature extractor
classification_model = joblib.load('model/classification_visual_only.pkl')
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_video_frames(video_path, interval=5):
    """Extract frames at every `interval` seconds."""
    video_clip = VideoFileClip(video_path)
    frames = []
    for t in range(0, int(video_clip.duration), interval):
        frame = video_clip.get_frame(t)
        frame_image = Image.fromarray(frame)
        frames.append(frame_image)
    return frames

def extract_visual_features(video_path):
    """Extract average ResNet features from video frames."""
    frames = extract_video_frames(video_path)
    frame_features = []

    for frame in frames:
        frame = frame.resize((224, 224))
        img_array = image.img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = resnet_model.predict(img_array)
        frame_features.append(features.flatten())

    return np.mean(frame_features, axis=0)

@app.route('/classify', methods=['POST'])
def classify_video():
    try:
        video_file = request.files['video']
        filename = video_file.filename
        temp_path = f'temp_{filename}'
        video_file.save(temp_path)

        visual_features = extract_visual_features(temp_path)
        prediction = classification_model.predict([visual_features])[0]

        os.remove(temp_path)
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
