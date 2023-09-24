from flask import Flask
import firebase_admin
from firebase_admin import credentials, storage
from dotenv import load_dotenv
import os

def create_app():
    # Load environment variables
    load_dotenv()

    # Flask initialisation
    app = Flask(__name__)

    app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 100 MB


    # Initialize Firebase
    cred = credentials.Certificate('./firebase-service-account.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
    })

    from .routes import upload, process, status, download, delete, start

    app.register_blueprint(upload.bp)
    app.register_blueprint(process.bp)
    app.register_blueprint(status.bp)
    app.register_blueprint(start.bp)
    app.register_blueprint(download.bp)
    app.register_blueprint(delete.bp)



    return app
