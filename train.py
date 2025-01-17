from data_loader import load_data
from feature_extractor import extract_features
from classifier import train_classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from datetime import datetime

def train_model():
    # Record the start time
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%H:%M:%S')}")

    # Load and preprocess data (with multithreading applied in data_loader.py)
    X, y = load_data()
    print("Data loaded successfully.")

    # Extract HOG features
    X_features = extract_features(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )

    # Parallelize decision tree processing within AdaBoost
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1, splitter="best"),  # splitter="best" supports multithreading
        n_estimators=50,
        algorithm="SAMME.R",
        learning_rate=0.5
    )

    # Train the classifier
    clf.fit(X_train, y_train)
    print("Classifier trained successfully.")

    # Save the trained model
    joblib.dump(clf, 'face_detection_model.joblib')
    print("Model saved to face_detection_model.joblib")

    # Record the end time
    end_time = datetime.now()
    print(f"Training finished at: {end_time.strftime('%H:%M:%S')}")

    # Calculate total training time
    training_time = end_time - start_time
    hours, remainder = divmod(training_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {hours} hrs, {minutes} minutes, {seconds} seconds")

if __name__ == "__main__":
    train_model()
