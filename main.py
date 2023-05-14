from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, storage
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask initialisation
app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate('./firebase-service-account.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})

#
# ROUTES
#


def upload_file(file, folder):
    filename = secure_filename(file.filename)
    blob = storage.bucket().blob(f'{folder}/{filename}')
    blob.upload_from_string(
        file.read(),
        content_type=file.content_type
    )
    return filename

@app.route('/api/upload/video/', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'no file'}), 400
    video = request.files['video']
    filename = upload_file(video, 'videos')
    return jsonify({'filename': filename}), 200

@app.route('/api/upload/img/', methods=['POST'])
def upload_img():
    if 'image' not in request.files:
        return jsonify({'error': 'no file'}), 400
    image = request.files['image']
    filename = upload_file(image, 'images')
    return jsonify({'filename': filename}), 200

@app.route('/api/process', methods=['POST'])
def process():
    data = request.get_json()
    if 'video_filename' not in data or 'image_filename' not in data:
        return jsonify({'error': 'missing filename'}), 400
    video_filename = data['video_filename']
    image_filename = data['image_filename']
    # The actual processing would likely be done in a background task,
    # but for simplicity, we'll assume it's done here.
    result_filename = process_video(video_filename, image_filename)
    return jsonify({'result_filename': result_filename}), 200

def process_video(video_filename, image_filename):
    # This is where you would implement your processing logic.
    # For now, we'll just return the video filename.
    return video_filename


# Run the Flask App
if __name__ == '__main__':
    app.run()