# pylint: disable=E1101
# pylint: disable=E0401

import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face_LBPHFaceRecognizer.create() # type: ignore
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_images_with_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    local_ids = []
    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        label = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(image_np)
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y+h, x:x+w])
            local_ids.append(label)
    return face_samples, local_ids

faces, label_ids = get_images_with_labels('DataSet')
recognizer.train(faces, np.array(label_ids))
recognizer.save('DataSet/training.yml')
