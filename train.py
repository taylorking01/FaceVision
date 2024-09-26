from data_loader import load_data
from feature_extractor import extract_features
from classifier import train_classifier
from sklearn.model_selection import train_test_split
from utils import plot_samples
import joblib

def train_model():
    # Load and preprocess data
    X, y = load_data()
    print("Data loaded successfully.")

    # Plot some samples
    plot_samples(X[:10], y[:10])

    # Extract features
    X_features = extract_features(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    # Train classifier
    clf = train_classifier(X_train, y_train)
    print("Classifier trained successfully.")

    # Evaluate classifier on test set
    accuracy = clf.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    joblib.dump(clf, 'face_detection_model.joblib')
    print("Model saved to face_detection_model.joblib")

if __name__ == "__main__":
    train_model()
