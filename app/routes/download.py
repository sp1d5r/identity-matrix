import json

from firebase_admin import storage
from flask import Blueprint, request, jsonify
import datetime


from app.services import firebase_service

bp = Blueprint('download', __name__, url_prefix='/api')

@bp.route('/download', methods=['GET'])
def status():
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
    video_name = job_data.get('video')

    print(video_name)

    # Construct the path to the file in Firebase Storage using the job_id
    # You might need to modify this depending on your actual path structure
    file_path_in_bucket = f'output/{video_name}'
    # Get the bucket
    bucket = storage.bucket(name='face-flipper.appspot.com')

    # Get the blob
    blob = bucket.blob(file_path_in_bucket)

    # Check if blob exists
    if not blob.exists():
        return jsonify({'error': 'File not found'}), 404

    # Define the expiration time for the signed URL
    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)  # URL will be valid for 15 minutes

    # Generate a download URL
    download_url = blob.generate_signed_url(expiration=expiration_time)

    # Return a JSON response with the download URL
    return jsonify({'download_url': download_url}), 200