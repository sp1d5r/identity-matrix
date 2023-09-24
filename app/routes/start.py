from flask import Blueprint, request, jsonify

from app.services import firebase_service

bp = Blueprint('start', __name__, url_prefix='/api')


@bp.route('/start/', methods=['POST'])
def start_job():
    # Check if both video and image files are part of the request
    if 'video' not in request.files or 'image' not in request.files:
        return jsonify({'error': 'Video and Image files are required'}), 400

    # Get the video and image files from the request
    video_file = request.files['video']
    image_file = request.files['image']

    # Define the folders where the video and image should be uploaded
    video_folder = 'videos'
    image_folder = 'images'

    # Upload the video file and get the video ID
    video_response, video_status_code = firebase_service.upload_file(request, 'video', video_folder)
    if video_status_code != 200:
        return video_response, video_status_code
    video_id = video_response.json['filename']

    # Upload the image file and get the image ID
    image_response, image_status_code = firebase_service.upload_file(request, 'image', image_folder)
    if image_status_code != 200:
        return image_response, image_status_code
    image_id = image_response.json['filename']

    # Create a job with the uploaded video and image IDs
    job_response = firebase_service.create_job(video_id, image_id)

    # Return the response from the create_job function
    return job_response

