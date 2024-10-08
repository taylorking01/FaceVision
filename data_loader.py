import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_lfw_people

def load_data():
    # Define data augmentation and preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to a consistent size
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),        # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])

    # Load non-face images (CIFAR-100)
    cifar100_train = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    cifar100_test = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )

    X_non_faces = [img[0] for img in cifar100_train] + [img[0] for img in cifar100_test]
    y_non_faces = [0] * len(X_non_faces)  # Label for non-face

    # Load Labeled Faces in the Wild (LFW) dataset for face images
    lfw_people = fetch_lfw_people(min_faces_per_person=0, resize=0.5, download_if_missing=True)
    X_faces = lfw_people.images
    y_faces = np.ones(X_faces.shape[0])

    # Transform LFW images
    face_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    X_faces_transformed = []
    for img in X_faces:
        img_tensor = face_transform(img.astype(np.uint8))
        X_faces_transformed.append(img_tensor)

    # Combine face and non-face images
    X = X_faces_transformed + X_non_faces
    y = np.concatenate((y_faces, y_non_faces), axis=0)

    return X, y
