from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def train_classifier(X_train, y_train):
    # Initialize AdaBoost with Decision Stump
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        algorithm="SAMME.R",
        learning_rate=0.5
    )
    # Train the classifier
    clf.fit(X_train, y_train)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    # Evaluate the classifier
    accuracy = clf.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
