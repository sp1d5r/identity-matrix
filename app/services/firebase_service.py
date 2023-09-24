from werkzeug.utils import secure_filename
from firebase_admin import storage, firestore
from flask import jsonify
import os
import uuid

db = firestore.client()

def upload_file(request, file_key, folder):
    # Check if the file is part of the request
    if file_key not in request.files:
        return jsonify({'error': 'no file'}), 400

    # Get the file from the request
    file = request.files[file_key]

    # Secure the original filename
    original_filename = secure_filename(file.filename)

    # Get the file extension of the original file
    _, file_extension = os.path.splitext(original_filename)

    # Generate a random unique ID for the filename
    unique_filename = str(uuid.uuid4()) + file_extension

    # Create a blob in the specified folder with the unique filename
    blob = storage.bucket().blob(f'{folder}/{unique_filename}')

    # Upload the file to the blob
    blob.upload_from_string(
        file.read(),
        content_type=file.content_type
    )

    # Return the unique filename in the response
    return jsonify({'filename': unique_filename}), 200


def get_job(job_id):
    # Get the reference to the job document
    job_ref = db.collection('jobs').document(job_id)

    # Retrieve the document
    job_doc = job_ref.get()

    # Check if the document exists
    if not job_doc.exists:
        # Return an error response if the job is not found
        return jsonify({'error': 'Job not found'}), 404

    # Retrieve the job data from the document
    job_data = job_doc.to_dict()

    # Return the job data in the response
    return jsonify(job_data), 200

def create_job(video_id, image_id):
    # Generate a random ID for the job
    job_id = str(uuid.uuid4())

    # Create a document in the Firestore collection for jobs
    jobs_ref = db.collection('jobs')
    job_ref = jobs_ref.add({
        'id': job_id,
        'video': video_id,
        'face_image': image_id,
        'progress': 0,
        'frame': 0,
        'stage': 'video_uploaded'
    })

    # Return the job ID in the response
    return jsonify({'job_id': job_id}), 200


def update_progress(job_id, progress, frame, stage):
    # Get the reference to the job document
    job_ref = db.collection('jobs').document(job_id)

    # Update the progress and stage of the job
    job_ref.update({
        'progress': progress,
        'frame': frame,
        'stage': stage
    })

    # Return a success response
    return jsonify({'success': True}), 200

def add_total_frames(job_id, total_frames):
    # Get the reference to the job document
    job_ref = db.collection('jobs').document(job_id)

    # Update the progress and stage of the job
    job_ref.update({
        'total_frames': total_frames,
    })

    # Return a success response
    return jsonify({'success': True}), 200