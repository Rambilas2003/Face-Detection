from mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from numpy.linalg import norm
facenet = load_model('facenet_keras.h5')
detector = MTCNN()

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return np.expand_dims(face, axis=0)

def get_embedding(face):
    face = preprocess_face(face)
    embedding = facenet.predict(face)
    return embedding[0]
img = cv2.imread('images/vk.jpg')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = detector.detect_faces(rgb_img)

for res in results:
    x, y, w, h = res['box']
    x, y = abs(x), abs(y)
    face = rgb_img[y:y+h, x:x+w]
    embedding = get_embedding(face)
    print("Embedding vector (first 5 dims):", embedding[:5])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for key, point in res['keypoints'].items():
        cv2.circle(img, point, 2, (0, 255, 0), 2)
cv2.imshow('Face with Landmarks & Embedding', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
