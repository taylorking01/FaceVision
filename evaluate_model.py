# evaluate_model.py
from feature_extractor import extract_features_single
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, Subset
from PIL import Image

def main():
    # Load the trained model
    clf = joblib.load('face_detection_model_svm.joblib')
    print("Model loaded successfully.")

    # Define transformation to match model's input
    transform = transforms.Compose([
        transforms.Resize((62, 47)),  # Ensure images are resized to (62, 47)
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Load face images from Olivetti Faces dataset
    olivetti = fetch_olivetti_faces()
    X_faces = olivetti.images
    y_faces = np.ones(len(X_faces))  # Label for face images

    # Ensure we have 500 face images
    num_face_samples = min(500, len(X_faces))
    X_faces = X_faces[:num_face_samples]
    y_faces = y_faces[:num_face_samples]

    # Resize face images to (62, 47) to match the model's expected input size
    X_faces_resized = []
    for img in X_faces:
        img_pil = Image.fromarray(np.uint8(img * 255))  # Convert to PIL Image
        img_resized = img_pil.resize((47, 62))  # PIL uses (width, height)
        img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
        X_faces_resized.append(img_array)
    X_faces = np.array(X_faces_resized)

    # Load non-face images from CIFAR-10
    cifar10_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Get 500 non-face images from CIFAR-10
    non_face_indices = []
    for idx, (img_tensor, label) in enumerate(cifar10_test):
        if label in [0, 1, 8, 9]:  # Classes: airplane, automobile, ship, truck
            non_face_indices.append(idx)
        if len(non_face_indices) >= 500:
            break
    cifar10_subset = Subset(cifar10_test, non_face_indices)
    non_face_loader = DataLoader(cifar10_subset, batch_size=1, shuffle=False)

    X_non_faces = []
    y_non_faces = []
    for img_tensor, _ in non_face_loader:
        img_array = img_tensor.numpy().squeeze()
        X_non_faces.append(img_array)
        y_non_faces.append(0)  # Label for non-face
    X_non_faces = np.array(X_non_faces)
    y_non_faces = np.array(y_non_faces)

    # Combine face and non-face images into a single NumPy array
    X_test = np.concatenate((X_faces, X_non_faces), axis=0)
    y_test = np.concatenate((y_faces, y_non_faces), axis=0)

    # Preprocess images and extract features using HOG
    X_test_processed = [extract_features_single(img) for img in X_test]
    X_test_processed = np.array(X_test_processed)
    print("Data preprocessing completed.")

    # Make predictions
    y_pred = clf.predict(X_test_processed)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on Unseen Data: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Face', 'Face']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
