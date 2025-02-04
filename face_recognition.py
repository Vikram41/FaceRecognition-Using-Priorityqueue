import numpy as np
import cv2
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet
import cv2
from scipy.spatial.distance import cosine

# Path to the Haar cascade XML file
face_cascade_path = 'haarcascades/haarcascade_frontalface_default.xml'

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


# For custom location, use the absolute path:
# face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Load the FaceNet model
model = FaceNet()

# Function to detect faces using OpenCV
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to align the face to the standard size (FaceNet input)
def align_face(face_image):
    aligned_face = cv2.resize(face_image, (160, 160))  # Resize to FaceNet input size
    aligned_face = np.expand_dims(aligned_face, axis=0)
    return aligned_face

# Function to generate embeddings for an image
def generate_embedding(face_image):
    aligned_face = align_face(face_image)
    embedding = model.embeddings(aligned_face)
    return embedding

# Function to compare two embeddings using cosine similarity
def compare_faces(embedding1, embedding2, threshold=0.6):
    # Flatten the embeddings to 1-D arrays
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    
    # Compare embeddings using cosine similarity
    similarity = cosine(embedding1, embedding2)
    if similarity < threshold:
        return True  # Faces match
    return False  # Faces don't match
