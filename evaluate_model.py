from feature_extractor import extract_features_single
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torchvision
import torchvision.transforms as transforms

def main():
    # Load the trained model
    clf = joblib.load('face_detection_model_svm.joblib')
    print("Model loaded successfully.")

    # Define transformation to match model's input
    transform = transforms.Compose([
        transforms.Resize((62, 47)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Load test face images from LFW (subset for testing)
    from sklearn.datasets import fetch_lfw_people
    lfw_test = fetch_lfw_people(
        min_faces_per_person=0, resize=0.5, download_if_missing=True
    )
    X_faces = lfw_test.images
    y_faces = np.ones(X_faces.shape[0])

    # Load non-face images from CIFAR-10
    cifar10_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Get a subset of CIFAR-10 images for non-face samples
    X_non_faces = []
    y_non_faces = []
    for img_tensor, label in cifar10_test:
        if label in [0, 1, 8, 9]:  # Select classes like airplane, automobile, ship, truck
            img_array = img_tensor.numpy().squeeze()
            X_non_faces.append(img_array)
            y_non_faces.append(0)  # Label for non-face
        if len(X_non_faces) >= 50:
            break

    X_non_faces = np.array(X_non_faces)
    y_non_faces = np.array(y_non_faces)

    # Use first 50 face images from LFW for testing
    X_humans = X_faces[:50]
    y_humans = y_faces[:50]

    # Combine human and non-human images into a single NumPy array
    X_test = np.concatenate((X_humans, X_non_faces), axis=0)
    y_test = np.concatenate((y_humans, y_non_faces), axis=0)

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
    print(classification_report(y_test, y_pred, target_names=['Non-Human', 'Human']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
