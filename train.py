import os
import numpy as np
from keras.preprocessing import image
from face_recognition import generate_embedding
import pickle

def create_embeddings(data_dir):
    embeddings = []
    names = []

    # Loop through each person in the dataset folder
    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                
                # Load and process image
                img = image.load_img(img_path, target_size=(160, 160))
                img_array = np.array(img)
                embedding = generate_embedding(img_array)
                
                # Store the embedding and the corresponding person name
                embeddings.append(embedding)
                names.append(person_name)
    
    # Save embeddings and names
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump((embeddings, names), f)

create_embeddings('data')
