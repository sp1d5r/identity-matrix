from flask import Blueprint, request, jsonify
from ..services.video_processing_service import process_video

bp = Blueprint('upload', __name__, url_prefix='/api/process')

@bp.route('/api/process', methods=['POST'])
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

