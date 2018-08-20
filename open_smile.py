import sys

from os import listdir
from os.path import isfile, join

import cv2
import dlib

from imutils import face_utils

from scipy.spatial import distance as dist


def evaluate_mouth_aspect_ratio(mouth_points):
    right_vertical_distance = dist.euclidean(mouth_points[3], mouth_points[9])
    mid_vertical_distance = dist.euclidean(mouth_points[2], mouth_points[10])
    left_vertical_distance = dist.euclidean(mouth_points[4], mouth_points[8])
    avg_vertical_distance = (right_vertical_distance + mid_vertical_distance + \
                             left_vertical_distance) / 3

    horizontal_distance = dist.euclidean(mouth_points[0], mouth_points[6])

    mouth_aspect_ratio = avg_vertical_distance / horizontal_distance

    return mouth_aspect_ratio


def detect_face(image, detector, blob_side=250, conf_threshold=0.6):
    im_height, im_width = image.shape[:2]

    rescale_factor_X = float(im_width) / blob_side
    rescale_factor_Y = float(im_height) / blob_side

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blob = cv2.resize(image_rgb, (blob_side, blob_side), cv2.INTER_NEAREST)

    detections = detector(blob, 1)

    if len(detections) == 0:
        return None

    rescaled_detection_points = [dlib.rectangle(
        int(d.rect.left() * rescale_factor_X),
        int(d.rect.top() * rescale_factor_Y),
        int(d.rect.right() * rescale_factor_X),
        int(d.rect.bottom() * rescale_factor_Y)) for _, d in enumerate(detections)
        if d.confidence >= conf_threshold]

    if len(rescaled_detection_points) == 0:
        return None

    return rescaled_detection_points


def process_dir(images_root="samples",
                detector_path="cnn_face.dat",
                predictor_path="shape_predictor_68_face_landmarks.dat"):
    cnn_face_detector = dlib.cnn_face_detection_model_v1(detector_path)
    predictor = dlib.shape_predictor(predictor_path)

    onlyfiles = [file for file in listdir(images_root) if isfile(join(images_root, file))]

    opened_mouth_file_list = []
    smile_mouth_file_list = []

    for file in onlyfiles:

        image_path = join(images_root, file)
        image = cv2.imread(image_path)

        detections = detect_face(image, cnn_face_detector)

        if not detections:
            continue

        landmarks_info = predictor(image, detections[0])
        landmarks = face_utils.shape_to_np(landmarks_info)
        mouth_landmarks = landmarks[49:60]
        mouth_aspect_ratio = evaluate_mouth_aspect_ratio(mouth_landmarks)

        if 0.8 > mouth_aspect_ratio > 0.5:
            smile_mouth_file_list.append(file)

        if mouth_aspect_ratio > 0.8:
            opened_mouth_file_list.append(file)

    print("Smile detected at: " + str(smile_mouth_file_list))
    print("Opened mouth detected at: " + str(opened_mouth_file_list))


if len(sys.argv) == 1:
    process_dir()
if len(sys.argv) == 2:
    process_dir(sys.argv[1])