# classifier.py
from sklearn.svm import SVC

def train_classifier(X_train, y_train):
    # Initialize the Support Vector Classifier (SVM)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can experiment with kernel, C, and gamma
    # Train the classifier
    clf.fit(X_train, y_train)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    # Evaluate the classifier
    accuracy = clf.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
