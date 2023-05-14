from flask import Blueprint, request, jsonify

bp = Blueprint('delete', __name__, url_prefix='/api')


@bp.route('/delete', methods=['GET'])
def status():
    job_id = request.args.get('job_id')
    if job_id is None:
        return jsonify({'error': 'job_id parameter is missing'}), 400

    # TODO: Get a download of the job id

    # Example response with the job_id
    return jsonify({'deleted_job': job_id}), 200