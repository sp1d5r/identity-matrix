from mtcnn import MTCNN
import cv2
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, storage
import numpy as np
import tempfile
import dotenv
from flask import Flask, request, jsonify
dotenv.load_dotenv()


# Initialize Firebase
cred = credentials.Certificate(json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT")))
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})

# Initialise the Flask app
app = Flask(__name__)

db = firestore.client()
bucket = storage.bucket()
def extract_faces(video_id):
    # Download the video from Firebase Storage
    print("Downloading file form firestore")
    video_blob = bucket.blob(f"videos/{video_id}.mp4")
    video_path = tempfile.mktemp(suffix=".mp4")
    video_blob.download_to_filename(video_path)
    print("Downloaded file form firestore")

    # Load the video
    print("Load the file")
    cap = cv2.VideoCapture(video_path)
    print("Load the file")

    # Initialize the face detector
    print("Initialising the face detector")
    detector = MTCNN()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    faces = {}

    # Initialise the firebase status and progress:
    print("Setting video frames")
    db.collection('extracted_frames').document(video_id).set({
        'preprocessing_progress': 0
    })

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print("Frame +1")
        frame_count += 1

        # Detect faces in the frame
        results = detector.detect_faces(frame)

        for i, result in enumerate(results):
            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height
            face = frame[y1:y2, x1:x2]

            # Save face image and position
            face_id = f"face_{i + 1}"
            if face_id not in faces:
                faces[face_id] = []
            faces[face_id].append({
                "frame": frame_count,
                "x": x1,
                "y": y1,
                "width": width,
                "height": height
            })

            # Convert the face image to bytes
            _, face_bytes = cv2.imencode('.jpg', face)
            face_bytes = face_bytes.tobytes()

            # Upload the face image to Firebase Storage
            blob = bucket.blob(f"videos/{video_id}/faces/{face_id}/frame_{frame_count}.jpg")
            blob.upload_from_string(face_bytes, content_type='image/jpg')

        # Update the progress in Firestore
        progress = frame_count / total_frames

        db.collection('extracted_frames').document(video_id).update({
            'preprocessing_progress': progress
        })

    cap.release()

    # Save the positional information to Firestore
    print("Saving positional here // think this is causing the problem")
    db.collection('extracted_frames').document(video_id).update({
        'positions': faces
    })


@app.route('/extract_faces', methods=['POST'])
def extract_faces_endpoint():
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        if video_id is None:
            return jsonify({'error': 'Missing or invalid video_id'}), 400

        extract_faces(video_id)
        return 'Faces extracted', 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))