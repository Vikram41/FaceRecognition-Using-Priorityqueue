import cv2
import numpy as np
from face_recognition import generate_embedding, compare_faces, detect_faces
import pickle
import heapq  # Import the heapq module for priority queue
from scipy.spatial.distance import cosine 
# Load known face embeddings
with open('embeddings.pkl', 'rb') as f:
    known_embeddings, known_names = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create an empty priority queue (min-heap)
priority_queue = []

# Define a function to add a face to the priority queue
def add_to_queue(face_name, similarity_score):
    # Push a tuple (similarity_score, face_name) to the priority queue
    # Lower similarity_score indicates a better match, so it gets higher priority in the queue
    heapq.heappush(priority_queue, (similarity_score, face_name))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        
        # Generate embedding for the detected face
        embedding = generate_embedding(face)
        
        # Compare with known embeddings and find the best match
        recognized = False
        for known_embedding, name in zip(known_embeddings, known_names):
            similarity = cosine(embedding.flatten(), known_embedding.flatten())
            
            if similarity < 0.6:  # Match threshold
                cv2.putText(frame, f"Hello {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add the recognized face to the priority queue with its similarity score
                add_to_queue(name, similarity)
                recognized = True
                break

        if not recognized:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Webcam Feed", frame)

    # Print the most recent recognized faces from the priority queue
    if priority_queue:
        # Get the most prioritized face (lowest cosine similarity, best match)
        most_similar_face = heapq.heappop(priority_queue)
        print(f"Most recognized face: {most_similar_face[1]}")
#with similarity: {most_similar_face[0]}
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
