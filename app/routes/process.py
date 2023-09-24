import json

from flask import Blueprint, request, jsonify
import os
import cv2
from firebase_admin import storage, firestore
from app.services import firebase_service
from app.services.face_swapping import swap_faces

bp = Blueprint('process', __name__, url_prefix='/api')

db = firestore.client()
bucket = storage.bucket()


@bp.route('/pre-process/', methods=['POST'])
def pre_process():
    data = request.get_json()

    # Validate job_id in the request
    job_id = data.get('job_id')
    if not job_id:
        return jsonify({'error': 'Job ID is required'}), 400

    # Retrieve the job information
    job_information, job_status_code = firebase_service.get_job(job_id)
    if job_status_code != 200:
        return job_information, job_status_code

    job_data = json.loads(job_information.get_data(as_text=True))

    # Extract video_id from the job information
    video_id = job_data.get('video')

    if not video_id:
        return jsonify({'error': 'Video ID not found in the job information'}), 400

    # Update the stage
    firebase_service.update_progress(job_id, progress=0, frame=0, stage="PREPROCESS")

    directory = "../assets"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the video from Firebase Storage
    video_blob = bucket.blob(f"videos/{video_id}")
    video_path = f"../assets/{video_id}"  # Specify a local path to save the video
    video_blob.download_to_filename(video_path)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame and upload it
        frame_filename = f"extracted_frames/{video_id}/frame_{frame_count}.png"
        frame_blob = bucket.blob(frame_filename)
        _, frame_bytes = cv2.imencode('.png', frame)
        frame_blob.upload_from_string(frame_bytes.tobytes(), content_type='image/png')

        # Update the progress
        progress = (frame_count + 1) / total_frames
        firebase_service.update_progress(job_id, progress=progress, stage="PREPROCESS")

    firebase_service.add_total_frames(total_frames)

    cap.release()

    return jsonify({'result_filename': "Success..."}), 200


@bp.route('/swap-frame/', methods=['POST'])
def swap_frame():
    data = request.get_json()

    # Validate job_id in the request
    job_id = data.get('job_id')
    if not job_id:
        return jsonify({'error': 'Job ID is required'}), 400

    # Retrieve the job information
    job_information, job_status_code = firebase_service.get_job(job_id)
    if job_status_code != 200:
        return job_information, job_status_code

    job_data = json.loads(job_information.get_data(as_text=True))
    stage = job_data.get('stage')
    video = job_data.get('video')
    image = job_data.get('face_image')
    total_frames = job_data.get('total_frames')

    current_frame = 0
    if stage != "PREPROCESS":
        current_frame = job_data.get('frame')

    for frame_count in range(current_frame, total_frames):
        res = swap_faces(video, frame_count, image)
        if res != 200:
            # Log error and stop
            print(f"Error swapping faces on frame {frame_count}")
            return jsonify({'error': f"Failed at frame {frame_count}"}), 400

        # Update the progress
        progress = (frame_count + 1) / total_frames
        firebase_service.update_progress(job_id, frame=frame_count, progress=progress, stage="FACE_SWAPPING")

    return jsonify({'result_filename': "Success..."}), 200
