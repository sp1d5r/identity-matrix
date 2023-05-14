from flask import Blueprint, request, jsonify
from app.services import firebase_service

bp = Blueprint('upload', __name__, url_prefix='/api/upload')

@bp.route('/video/', methods=['POST'])
def upload_video():
    return firebase_service.upload_file(request, 'video', 'videos')

@bp.route('/img/', methods=['POST'])
def upload_img():
    return firebase_service.upload_file(request, 'image', 'images')