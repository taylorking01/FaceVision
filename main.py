from feature_extractor import extract_features_single
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torchvision
import torchvision.transforms as transforms

def main():
    # Load the trained model
    clf = joblib.load('face_detection_model.joblib')
    print("Model loaded successfully.")

    # Define transformation to match model's input
    transform = transforms.Compose([
        transforms.Resize((62, 47)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Load CelebA test dataset (Alternative: Use a different dataset if CelebA remains inaccessible)
    # We'll use the Test Split of LFW instead to avoid downloading issues
    from sklearn.datasets import fetch_lfw_people

    lfw_test = fetch_lfw_people(
        min_faces_per_person=0,
        resize=0.5,
        download_if_missing=True,
    )

    X_faces = lfw_test.images
    y_faces = np.ones(X_faces.shape[0])

    # Generate non-human images using CIFAR-10
    cifar10 = torchvision.datasets.CIFAR10(
        root='/content',
        train=False,
        download=True,
        transform=transform,
    )

    def get_non_human_images(dataset, num_samples=50):
        non_human_classes = [0, 1, 8, 9]  # 'airplane', 'automobile', 'ship', 'truck'
        images = []
        count = 0
        i = 0
        while count < num_samples and i < len(dataset):
            img_tensor, label = dataset[i]
            if label in non_human_classes:
                img_array = img_tensor.numpy().squeeze()
                images.append(img_array)
                count += 1
            i += 1
        return images

    non_human_images = get_non_human_images(cifar10)
    y_non_humans = np.zeros(len(non_human_images))

    # Assign labels
    X_humans = X_faces[:50]  # Select first 50 face images
    y_humans = y_faces[:50]

    # Combine human and non-human images into a single NumPy array
    X_test = np.concatenate((X_humans, non_human_images), axis=0)  # Correctly concatenate along axis 0
    y_test = np.concatenate((y_humans, y_non_humans), axis=0)

    # Debugging: Print shapes
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Preprocess images
    X_test_processed = [extract_features_single(img) for img in X_test]
    X_test_processed = np.array(X_test_processed)
    print(f"X_test_processed shape: {X_test_processed.shape}")
    print("Data preprocessing completed.")

    # Make predictions
    y_pred = clf.predict(X_test_processed)

    # Debugging: Print y_pred shape
    print(f"y_pred shape: {y_pred.shape}")

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on Unseen Data: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Human', 'Human']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
