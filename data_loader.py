import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_lfw_people

def load_data():
    # Define transformations to resize images and convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((62, 47)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Load CIFAR-100 dataset for non-face images
    cifar100_train = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    cifar100_test = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )

    # Convert CIFAR-100 dataset to arrays
    X_non_faces = []
    y_non_faces = []
    for img_tensor, _ in cifar100_train + cifar100_test:
        img_array = img_tensor.numpy().squeeze()
        X_non_faces.append(img_array)
        y_non_faces.append(0)  # Label for non-face

    X_non_faces = np.array(X_non_faces)
    y_non_faces = np.array(y_non_faces)

    # Load Labeled Faces in the Wild (LFW) dataset for face images
    lfw_people = fetch_lfw_people(min_faces_per_person=0, resize=0.5, download_if_missing=True)
    X_faces = lfw_people.images
    y_faces = np.ones(X_faces.shape[0])

    # Combine face and non-face images
    X = np.concatenate((X_faces, X_non_faces), axis=0)
    y = np.concatenate((y_faces, y_non_faces), axis=0)

    return X, y
