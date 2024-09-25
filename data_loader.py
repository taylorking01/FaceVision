import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.utils import shuffle
from skimage.transform import resize

def load_data():
    # Fetch LFW faces dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=0, resize=1.0, download_if_missing=True)
    X_faces = lfw_people.images
    y_faces = np.ones(X_faces.shape[0])

    # Generate negative samples (e.g., random noise)
    X_non_faces = np.random.rand(1000, 62, 47)
    y_non_faces = np.zeros(X_non_faces.shape[0])

    # Combine datasets
    X = np.concatenate((X_faces, X_non_faces), axis=0)
    y = np.concatenate((y_faces, y_non_faces), axis=0)

    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)

    # Normalize the images
    X = X / 255.0

    return X, y
