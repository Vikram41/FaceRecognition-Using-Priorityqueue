import cv2
import numpy as np
from face_recognition import generate_embedding, compare_faces, detect_faces
import pickle
import heapq
from scipy.spatial.distance import cosine
import time
import matplotlib.pyplot as plt

# Load known face embeddings
with open('embeddings.pkl', 'rb') as f:
    known_embeddings, known_names = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create an empty priority queue (min-heap)
priority_queue = []

# Metrics to track performance
total_response_time_with_queue = 0
total_response_time_without_queue = 0
face_count_with_queue = 0
face_count_without_queue = 0
frames = []

frame_count = 0
total_frames = 50  # Set the number of frames to evaluate

while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    faces = detect_faces(frame)

    # Measure response time without data structure
    start_time_no_ds = time.time()
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        embedding = generate_embedding(face)
        for known_embedding, name in zip(known_embeddings, known_names):
            similarity = cosine(embedding.flatten(), known_embedding.flatten())
            if similarity < 0.6:
                face_count_without_queue += 1
    end_time_no_ds = time.time()
    total_response_time_without_queue += (end_time_no_ds - start_time_no_ds)

    # Measure response time with priority queue
    start_time_with_ds = time.time()
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        embedding = generate_embedding(face)
        for known_embedding, name in zip(known_embeddings, known_names):
            similarity = cosine(embedding.flatten(), known_embedding.flatten())
            if similarity < 0.6:
                heapq.heappush(priority_queue, (similarity, name))
                face_count_with_queue += 1
    end_time_with_ds = time.time()
    total_response_time_with_queue += (end_time_with_ds - start_time_with_ds)

    # Break after processing the desired number of frames
    if frame_count >= total_frames:
        break

cap.release()
cv2.destroyAllWindows()

# Prepare data for bar graph
response_time_data = [total_response_time_with_queue / frame_count,
                      total_response_time_without_queue / frame_count]
face_count_data = [face_count_with_queue, face_count_without_queue]

# Plot bar graph for response times
plt.figure(figsize=(12, 6))
plt.bar(["With Priority Queue", "Without Priority Queue"], response_time_data, color=['blue', 'red'])
plt.title("Average Response Time Per Frame")
plt.ylabel("Response Time (seconds)")
plt.show()

# Plot bar graph for face counts
plt.figure(figsize=(12, 6))
plt.bar(["With Priority Queue", "Without Priority Queue"], face_count_data, color=['green', 'orange'])
plt.title("Number of Faces Processed")
plt.ylabel("Face Count")
plt.show()
