import cv2
import insightface
from insightface.app import FaceAnalysis
from firebase_admin import storage
import numpy as np
import os

# Initialize the model
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id=0, det_size=(640,640))
swapping_model = insightface.model_zoo.get_model('/Users/elijahahmad/PycharmProjects/Identity Matrix/app/assets/inswapper_128_fp16.onnx', download=False, download_zip=False)



# Define file paths
frame_filename = f"/Users/elijahahmad/PycharmProjects/Identity Matrix/app/ml/face_swapping_2/assets/input/img.png"
driving_path = f"/Users/elijahahmad/PycharmProjects/Identity Matrix/app/ml/face_swapping_2/assets/input/img_1.png"
output_path = f"/Users/elijahahmad/PycharmProjects/Identity Matrix/app/ml/face_swapping_2/assets/output"

try:
    # Download source image (frame) from Firebase
    frame = cv2.imread(frame_filename) # from source

    # Download driving image from Firebase
    driving_img = cv2.imread(driving_path)

    # Detect faces in the target (frame) and source (driving image) images
    target_faces = app.get(frame)
    source_faces = app.get(driving_img)

    if len(source_faces) == 0:
        print("Failed to find face in driving image.")


    source_face = source_faces[0]

    frame_copy = frame.copy()

    for face in target_faces:
        frame_copy = swapping_model.get(frame_copy, face, source_face, paste_back=True)

    # Encode the swapped image to .png format
    _, swapped_img_encoded = cv2.imencode('.png', frame_copy)

    # Upload the swapped image back to Firebase
    # save to files


    # Create the full path to the output file
    output_filename = 'output_image.png'
    output_filepath = os.path.join("/Users/elijahahmad/PycharmProjects/Identity Matrix/app/ml/face_swapping_2/assets/output", output_filename)
    cv2.imwrite(output_filepath, frame_copy)

except Exception as e:
    print(f"Error: {e}")

