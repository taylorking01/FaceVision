import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_lfw_people
from concurrent.futures import ThreadPoolExecutor

def preprocess_image(img_tensor):
    """ Preprocess a single image: convert to numpy and squeeze to the expected shape. """
    return img_tensor.numpy().squeeze()

def load_data():
    # Define data augmentation techniques for face and non-face images
    transform = transforms.Compose([
        transforms.Resize((62, 47)),  # Resize to match model input size
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
        transforms.RandomRotation(degrees=30),  # Randomly rotate images
        transforms.RandomResizedCrop(size=(62, 47), scale=(0.8, 1.0)),  # Random scaling
        transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Adjust brightness and contrast
        transforms.ToTensor()  # Convert to tensor
    ])

    # Load CIFAR-100 dataset for non-face images
    cifar100_train = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    cifar100_test = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )

    # Use multithreading for processing the CIFAR-100 dataset
    with ThreadPoolExecutor() as executor:
        X_non_faces = list(executor.map(preprocess_image, [img[0] for img in cifar100_train + cifar100_test]))
        y_non_faces = [0] * len(X_non_faces)  # Label for non-face

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
