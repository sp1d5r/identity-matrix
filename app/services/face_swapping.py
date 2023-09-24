import cv2
import insightface
from insightface.app import FaceAnalysis
from firebase_admin import storage
import numpy as np
import os

# Use Cloud GPU


# Initialize the model
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id=0, det_size=(640,640))
swapping_model = insightface.model_zoo.get_model('/Users/elijahahmad/PycharmProjects/Identity Matrix/app/assets/inswapper_128_fp16.onnx', download=False, download_zip=False)

def swap_faces(video, frame_number, driving_image):
    # Define Firebase storage bucket
    bucket = storage.bucket()

    # Define file paths
    frame_filename = f"extracted_frames/{video}/frame_{frame_number}.png"
    driving_path = f"images/{driving_image}"
    output_path = f"swapped_frames/{video}/frame_{frame_number}.png"

    try:
        # Download source image (frame) from Firebase
        frame_blob = bucket.blob(frame_filename)
        frame = cv2.imdecode(np.frombuffer(frame_blob.download_as_bytes(), np.uint8), -1)

        # Download driving image from Firebase
        driving_blob = bucket.blob(driving_path)
        driving_img = cv2.imdecode(np.frombuffer(driving_blob.download_as_bytes(), np.uint8), -1)

        # Ensure frame is in 3 channels (RGB)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Ensure driving_img is in 3 channels (RGB)
        if driving_img.shape[2] == 4:
            driving_img = cv2.cvtColor(driving_img, cv2.COLOR_BGRA2BGR)

        # Detect faces in the target (frame) and source (driving image) images
        target_faces = app.get(frame)
        source_faces = app.get(driving_img)

        if len(source_faces) == 0:
            print("Failed to find face in driving image.")
            return 400

        source_face = source_faces[0]

        frame_copy = frame.copy()

        for face in target_faces:
            frame_copy = swapping_model.get(frame_copy, face, source_face, paste_back=True)

        # Encode the swapped image to .png format
        _, swapped_img_encoded = cv2.imencode('.png', frame_copy)

        # Upload the swapped image back to Firebase
        output_filename = 'output_image.png'
        output_filepath = os.path.join(
            "/Users/elijahahmad/PycharmProjects/Identity Matrix/app/ml/face_swapping_2/assets/output", output_filename)
        cv2.imwrite(output_filepath, frame_copy)

        output_blob = bucket.blob(output_path)
        output_blob.upload_from_string(swapped_img_encoded.tobytes(), content_type='image/png')

        return 200
    except Exception as e:
        print(f"Error: {e}")
        return 400
