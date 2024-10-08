import torch
from cnn_model import FaceDetectionCNN
from data_loader import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def main():
    # Load the trained model
    model = FaceDetectionCNN()
    model.load_state_dict(torch.load('face_detection_cnn.pth'))
    model.eval()
    print("Model loaded successfully.")

    # Load data
    X, y = load_data()
    print("Data loaded successfully.")

    # Convert lists to tensors
    X = torch.stack(X)
    y = torch.tensor(y).long()

    # Split data (use the same test set as in training)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluate the model
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

    # Calculate metrics
    accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test.numpy(), predicted.numpy(), target_names=['Non-Face', 'Face']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test.numpy(), predicted.numpy()))

if __name__ == "__main__":
    main()
