import cv2
import numpy as np
import dlib
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import os
import json
import dotenv
from flask import Flask, request, jsonify

dotenv.load_dotenv()


# Initialize Firebase
cred = credentials.Certificate(json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT")))
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})

# Initialise the Flask app
app = Flask(__name__)

# Get a reference to the storage bucket
bucket = storage.bucket()

# Load the detector and predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def swap_faces(img1, img2):
    # Detect faces in both images
    faces1 = detector(img1)
    faces2 = detector(img2)

    # Make sure both images have exactly one face detected
    if len(faces1) != 1 or len(faces2) != 1:
        print("A face was not found img1:", len(faces1), " img2:", len(faces2))
        return img1, img2

    # Get the facial landmarks for both faces
    landmarks1 = predictor(img1, faces1[0])
    landmarks2 = predictor(img2, faces2[0])

    # Select the points for the regions of interest (ROIs)
    roi_pts1 = np.array([(landmarks1.part(n).x, landmarks1.part(n).y) for n in range(17, 68)])
    roi_pts2 = np.array([(landmarks2.part(n).x, landmarks2.part(n).y) for n in range(17, 68)])

    # Calculate the convex hull for the ROIs
    hull1 = cv2.convexHull(roi_pts1)
    hull2 = cv2.convexHull(roi_pts2)

    # Find the bounding rectangles for the hulls
    rect1 = cv2.boundingRect(hull1)
    rect2 = cv2.boundingRect(hull2)

    # Crop the corresponding regions from both images
    roi_img1 = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    roi_img2 = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

    # Resize the cropped regions to the same size
    roi_img2_resized = cv2.resize(roi_img2, (roi_img1.shape[1], roi_img1.shape[0]))

    # Swap the faces by replacing the ROIs
    img1_swap = img1.copy()
    img1_swap[rect1[1]:rect1[1] + roi_img1.shape[0], rect1[0]:rect1[0] + roi_img1.shape[1]] = roi_img2_resized
    img2_swap = img2.copy()
    img2_swap[rect2[1]:rect2[1] + roi_img2_resized.shape[0], rect2[0]:rect2[0] + roi_img2_resized.shape[1]] = roi_img1

    return img1_swap, img2_swap


def resize_and_pad(image, target_size):
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the scaling factor for resizing
    scale = min(target_width / width, target_height / height)

    # Resize the image to the target size while maintaining aspect ratio
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    # Create a blank canvas of the target size with black borders
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    padded_height, padded_width = padded_image.shape[:2]

    # Calculate the offset to center the resized image
    x_offset = (padded_width - resized_image.shape[1]) // 2
    y_offset = (padded_height - resized_image.shape[0]) // 2

    # Copy the resized image onto the padded canvas
    padded_image[y_offset:y_offset+resized_image.shape[0], x_offset:x_offset+resized_image.shape[1]] = resized_image

    return padded_image


def remove_padding(image):
    # Find the non-zero regions of the image
    coords = cv2.findNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # Get the bounding rectangle for the non-zero regions
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the image to the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image


def check_image_dimensions(image1, image2):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    if height1 == height2 and width1 == width2:
        print(f"The images have the same dimensions. image1 {image1.shape} and image2 {image2.shape}")
    else:
        print(f"The images have the different dimensions. image1 {image1.shape} and image2 {image2.shape}")


def load_image_from_path(input_path):
    # Get a blob reference to the image file
    blob = bucket.blob(input_path)

    # Download the image file as bytes
    image_bytes = blob.download_as_bytes()

    # Convert the image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the image array using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image


def upload_image_to_storage(image, destination_path):
    # Create a blob object with the desired destination path
    blob = bucket.blob(destination_path)

    # Convert the image to bytes
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

    # Upload the image bytes to the blob
    blob.upload_from_string(image_bytes, content_type='image/jpeg')

    print(f"Image uploaded successfully to {destination_path}")


def swap_face_images(input_path1, input_path2, output_path):

    # Load the input images
    print("Looking at paths ", input_path1, input_path2)
    img1 = load_image_from_path(input_path1)
    img2 = load_image_from_path(input_path2)
    print("Downloaded images")

    # Determine the target size based on the larger image dimensions
    target_size = (max(img1.shape[1], img2.shape[1]), max(img1.shape[0], img2.shape[0]))

    # Resize and pad both images to the target size
    img1_resized = resize_and_pad(img1, target_size)
    img2_resized = resize_and_pad(img2, target_size)

    # Perform face swapping
    img1_swap, img2_swap = swap_faces(img1_resized, img2_resized)

    # Remove the padding from the swapped images
    img1_swap = remove_padding(img1_swap)
    # img2_swap = remove_padding(img2_swap)

    # Example usage
    output_image = img1_swap  # Replace with your output image
    upload_image_to_storage(output_image, output_path)

    # Print a message to indicate that the image has been saved
    print("Output image saved to:", output_path)


def swap_faces_for_frame(target_video, source_image, face_id, frame_number):
    swap_face_images(
        f"videos/{target_video}/faces/{face_id}/frame_{frame_number}.jpg",
        source_image,
        f"face_swapped/{target_video}/faces/{face_id}/frame_{frame_number}.jpg"
    )


swap_faces_for_frame("CrowPoint_-_Lie_-_4", "images/test-image-2.png", "face_1", 100)

@app.route('/swap_faces', methods=['POST'])
def extract_faces_endpoint():
    try:
        data = request.get_json()
        target_video = data.get('target_video')
        source_image = data.get('source_image')
        face_id = data.get('face_id')
        frame_number = data.get("frame_number")
        if target_video is None:
            return jsonify({'error': 'Missing or invalid video_id'}), 400
        if source_image is None:
            return jsonify({'error': 'Missing or invalid source_image'}), 400
        if face_id is None:
            return jsonify({'error': 'Missing or invalid face_id'}), 400
        if frame_number is None:
            return jsonify({'error': 'Missing or invalid frame_number'}), 400

        swap_faces_for_frame(target_video, source_image, face_id, frame_number)

        return f'Faces swapped! new path:face_swapped/{target_video}/faces/{face_id}/frame_{frame_number}.jpg', 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))