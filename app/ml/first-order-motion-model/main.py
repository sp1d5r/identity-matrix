import torch
import cv2
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import dlib
from models.generator import OcclusionAwareGenerator
from models.keypoint_detector import KPDetector
from models.data_loader import Logger
import yaml
import imageio

# Load the YAML file
with open('vox-adv-cpk.yml', 'r') as f:
    config = yaml.safe_load(f)

# Access the configuration parameters
dataset_params = config['dataset_params']
model_params = config['model_params']
train_params = config['train_params']
reconstruction_params = config['reconstruction_params']
animate_params = config['animate_params']
visualizer_params = config['visualizer_params']

# # Load pre-trained weights
# checkpoint_path = "./vox-adv-cpk.pth.tar"  # Path to pre-trained weights
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# generator = checkpoint['generator']
# kp_detector = checkpoint['kp_detector']
# print(kp_detector.keys())
# print(checkpoint.keys())

generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

checkpoint_path = "./vox-adv-cpk.pth.tar"  # Path to pre-trained weights
Logger.load_cpk(checkpoint_path, generator=generator, kp_detector=kp_detector)

# Check the model weights.
for name, param in generator.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"NaN or inf found in generator weights: {name}")
    else:
        print("Generator Model weights loaded Properly!")

for name, param in kp_detector.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"NaN or inf found in kp_detector weights: {name}")
    else:
        print("KPDetector Model weights loaded Properly!")


# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Define variables for temporal smoothing
prev_face_rect = None
SMOOTHING_FACTOR = 0.6  # Adjust this value between 0.0 and 1.0 for desired smoothing
FACE_PADDING = 25


def extract_face(image, face_padding):
    # We will be using temporarl smoothing to extract the face nicely.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    # Extract the first face
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Temporal smoothing
    global prev_face_rect
    if prev_face_rect is not None:
        x = int(x * SMOOTHING_FACTOR + prev_face_rect[0] * (1 - SMOOTHING_FACTOR))
        y = int(y * SMOOTHING_FACTOR + prev_face_rect[1] * (1 - SMOOTHING_FACTOR))
        w = int(w * SMOOTHING_FACTOR + prev_face_rect[2] * (1 - SMOOTHING_FACTOR))
        h = int(h * SMOOTHING_FACTOR + prev_face_rect[3] * (1 - SMOOTHING_FACTOR))

    # Calculate square bounding box
    size = max(w, h)
    cx = x + w // 2
    cy = y + h // 2
    x = cx - size // 2 - face_padding
    y = cy - size // 2 - face_padding
    size = size + face_padding * 2

    # Extract face region
    face_img = image[y:y + size, x:x + size]

    # Update previous face rectangle
    prev_face_rect = (x, y, size, size)

    return face_img


def crop_driving_video(driving_video_path, output_path):
    # Read driving video frames
    cap = cv2.VideoCapture(driving_video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract and preprocess face from each frame
        face_img = extract_face(frame, FACE_PADDING)
        if face_img is not None:
            face_img = resize(face_img, (256, 256))
            frames.append(face_img)

    cap.release()

    # Save the cropped frames as a video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    for frame in frames:
        frame = img_as_ubyte(frame.clip(0, 1))
        video_writer.write(frame)
    video_writer.release()

    print("Cropping completed!")


# Example usage Cropping Driving Video
driving_video_path = "./assets/driving-video-uncropped.mp4"
output_path = "./assets/driving-video-cropped.mp4"

# crop_driving_video(driving_video_path, output_path)

'''

Animate the first photo.

'''

def check_kp_dict(kp_dict, dict_name):
    for key, value in kp_dict.items():
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"NaN or inf found in {dict_name} for key: {key}")


def make_animation(target_image, source_frames, generator, kp_detector):
    predictions = []
    with torch.no_grad():
        for i, source_frame in enumerate(source_frames):
            source_frame = source_frame.unsqueeze(0)
            driving = torch.cat([source_frame] * len(target_image))

            # Extract keypoints from source and target images
            kp_source = kp_detector.forward(source_frame)
            kp_driving_initial = kp_detector.forward(driving)
            # Perform keypoint adaptation
            kp_driving = align_keypoints(kp_source, kp_driving_initial)

            check_kp_dict(kp_source, 'kp_source')
            check_kp_dict(kp_driving_initial, 'kp_driving_initial')
            check_kp_dict(kp_driving, 'kp_driving')

            # Generate the animation
            out = generator(source_frame, kp_source=kp_source, kp_driving=kp_driving)

            # Append the generated frame to predictions
            pred = out['prediction'].squeeze(0)
            predictions.append(pred)

            # Convert the prediction to a uint8 image
            pred_img = (pred.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # Save the image
            imageio.imsave(f'./assets/predictions/prediction_{i}.png', pred_img)
            print("saved img")

    return predictions


def align_keypoints(kp_source, kp_driving_initial):
    # Perform keypoint adaptation/alignment
    kp_norm = normalize_keypoints(kp_source)
    kp_driving = denormalize_keypoints(kp_norm, kp_driving_initial)

    return kp_driving

def normalize_keypoints(kp):
    if isinstance(kp, dict):
        norm_kp = {}
        for name, kp_tensor in kp.items():
            if not isinstance(kp_tensor, torch.Tensor):
                raise TypeError(f"Expected tensor for keypoint '{name}', but got {type(kp_tensor)}")
            norm_kp[name] = kp_tensor.clone()
            norm_kp[name][..., :2] = 2.0 * (norm_kp[name][..., :2] / norm_kp[name].new(norm_kp[name].shape[-2:]).float()) - 1.0
    else:
        if not isinstance(kp, torch.Tensor):
            raise TypeError(f"Expected tensor for keypoints, but got {type(kp)}")
        norm_kp = kp.clone()
        norm_kp[..., :2] = 2.0 * (norm_kp[..., :2] / norm_kp.new(norm_kp.shape[-2:]).float()) - 1.0
    return norm_kp


def denormalize_keypoints(norm_kp, kp_initial):
    if isinstance(norm_kp, dict):
        kp = {}
        for name, kp_tensor in norm_kp.items():
            if not isinstance(kp_tensor, torch.Tensor):
                raise TypeError(f"Expected tensor for normalized keypoint '{name}', but got {type(kp_tensor)}")
            kp[name] = kp_tensor.clone()
            kp[name][..., :2] = (kp[name][..., :2] + 1.0) * kp_initial[name].new(kp_initial[name].shape[-2:]).float() / 2.0
    else:
        if not isinstance(norm_kp, torch.Tensor):
            raise TypeError(f"Expected tensor for normalized keypoints, but got {type(norm_kp)}")
        kp = norm_kp.clone()
        kp[..., :2] = (kp[..., :2] + 1.0) * kp_initial.new(kp_initial.shape[-2:]).float() / 2.0
    return kp



def animate_image(source_video_path, target_image_path, output_path):
    # Read source video frames
    cap = cv2.VideoCapture(source_video_path)
    source_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract and preprocess face from each frame
        face_img = frame
        if face_img is not None:
            face_img = resize(face_img, (256, 256))
            source_frames.append(face_img)

    cap.release()

    # Read target image
    target_image = cv2.imread(target_image_path)

    target_image = extract_face(target_image, 50)
    # Preprocess target image
    target_image = resize(target_image, (256, 256))

    # Print the shape of target_image
    print(f"Shape of target_image before squeeze: {target_image.shape}")

    # Convert the target image to a uint8 image
    if target_image.ndim > 3:
        target_image = target_image.squeeze(0)

    # Print the shape of target_image before transpose
    print(f"Shape of target_image before transpose: {target_image.shape}")

    if target_image.shape[0] == 3:
        # Image is in (channels, height, width) format, transpose the axes
        target_image = target_image.transpose(1, 2, 0)

    print(f"Shape of target_image after transpose: {target_image.shape}")

    target_image_img = (target_image * 255).astype(np.uint8)

    print(f"Shape of target_image_img: {target_image_img.shape}")
    imageio.imsave('assets/test-image-cropped.png', target_image_img)

    target_image = torch.tensor(target_image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

    # Preprocess source video frames
    source_frames = np.array(source_frames)
    source_frames = torch.tensor(source_frames.transpose(0, 3, 1, 2)).float() / 255.0

    # Print out some information about the source_frames
    print(f"Number of frames: {len(source_frames)}")
    print(f"Shape of a frame: {source_frames[0].shape}")
    print(f"Data type of a frame: {source_frames[0].dtype}")

    # Check if there are any NaN or Inf values in the frames
    if torch.isnan(target_image).any() or torch.isinf(target_image).any():
        print("NaN or Inf values found in the frames")
    else:
        print("No NaN or Inf values found in the frames")

    if torch.isnan(target_image).any() or torch.isinf(target_image).any():
        print("NaN or inf found in target_image")
    else:
        print("Preprocessed Target Image is fine")

    # Run the animation
    predictions = make_animation(target_image, source_frames, generator, kp_detector)

    # Save the animated frames as a video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = predictions[0].shape[:2]
    video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    for pred in predictions:
        pred = img_as_ubyte(pred.clip(0, 1))
        video_writer.write(pred)
    video_writer.release()

    print("Animation completed!")

# Example usage
source_video_path = "./assets/driving-video-cropped.mp4"
target_image_path = "./assets/test-image.png"
output_path = "./assets/test-image-animated"

animate_image(source_video_path, target_image_path, output_path)