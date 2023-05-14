from werkzeug.utils import secure_filename
from firebase_admin import storage
from flask import jsonify


def upload_file(request, file_key, folder):
    if file_key not in request.files:
        return jsonify({'error': 'no file'}), 400
    file = request.files[file_key]
    filename = secure_filename(file.filename)
    blob = storage.bucket().blob(f'{folder}/{filename}')
    blob.upload_from_string(
        file.read(),
        content_type=file.content_type
    )
    return jsonify({'filename': filename}), 200
