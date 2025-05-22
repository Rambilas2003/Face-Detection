from mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from numpy.linalg import norm
import os
facenet = load_model('facenet_keras.h5')
detector = MTCNN()

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return np.expand_dims(face, axis=0)

def get_embedding(face):
    return facenet.predict(preprocess_face(face))[0]

def is_match(known_emb, test_emb, threshold=0.6):
    return norm(known_emb - test_emb) < threshold

def load_known_faces(folder='images/known_faces'):
    database = {}
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        if faces:
            x, y, w, h = faces[0]['box']
            face = rgb[y:y+h, x:x+w]
            name = os.path.splitext(file)[0]
            database[name] = get_embedding(face)
    return database
known_faces = load_known_faces()
img = cv2.imread('images/vk.jpg')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = detector.detect_faces(rgb_img)

correct_matches = 0
total_faces = len(results)

for res in results:
    x, y, w, h = res['box']
    x, y = abs(x), abs(y)
    face = rgb_img[y:y+h, x:x+w]
    embedding = get_embedding(face)
    name = "Unknown"
    min_dist = 100

    for person, known_emb in known_faces.items():
        dist = norm(known_emb - embedding)
        if dist < min_dist and dist < 0.6:
            min_dist = dist
            name = person
    if name != "Unknown":
        correct_matches += 1
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for point in res['keypoints'].values():
        cv2.circle(img, point, 2, (255, 0, 0), 2)
accuracy = (correct_matches / total_faces * 100) if total_faces > 0 else 0
print(f"Recognition Accuracy: {accuracy:.2f}%")
cv2.imshow("Face Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
