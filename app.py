import os
from flask import Flask, request, jsonify
from keras.preprocessing.image import img_to_array
from keras.backend import clear_session
import imutils
import cv2
from keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
UPLOAD_FOLDER = 'media'
face_detection = cv2.CascadeClassifier(detection_model_path)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def process(img):
    clear_session()
    emotion_classifier = load_model(emotion_model_path, compile=False)
    frame = cv2.imread(UPLOAD_FOLDER+"/"+img, cv2.IMREAD_GRAYSCALE)
    frame = imutils.resize(frame, width=300)
    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:

        for ele in faces:
            (x, y, w, h) = ele
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]

    emotion={
        "angry": str(preds[0]),
        "disgust": str(preds[1]),
        "scared": str(preds[2]),
        "happy": str(preds[3]),
        "sad": str(preds[4]),
        "surprised": str(preds[5]),
        "neutral": str(preds[6])
    }

    return jsonify(emotion)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/data/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            emotions = process(filename)
            return emotions, 201
    return


@app.route('/', methods=['GET'])
def hello():
    return "hello",201



if __name__ == '__main__':
    app.run()
