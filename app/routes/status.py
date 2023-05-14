from flask import Blueprint, request, jsonify

bp = Blueprint('status', __name__, url_prefix='/api')


@bp.route('/status', methods=['GET'])
def status():
    job_id = request.args.get('job_id')
    if job_id is None:
        return jsonify({'error': 'job_id parameter is missing'}), 400

    # TODO: Get the status of the job with the provided job_id

    # Example response with the job_id
    return jsonify({'job_id': job_id}), 200