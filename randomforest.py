import util
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import pandas as pd

def main():
    train_path = "train_split.csv"
    test_path = "test_split.csv"
    train_x, train_y = util.load_dataset(train_path, add_intercept=False)
    test_x, test_y = util.load_dataset(test_path, add_intercept=False)

    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_classifier.fit(train_x, train_y)

    rf_train_predictions = rf_classifier.predict(train_x)
    rf_test_predictions = rf_classifier.predict(test_x)

    # Training Accuracy
    rf_train_accuracy = accuracy_score(train_y, rf_train_predictions)
    print("Random Forest Training Accuracy:", rf_train_accuracy)

    # Testing Accuracy
    rf_test_accuracy = accuracy_score(test_y, rf_test_predictions)
    print("Random Forest Testing Accuracy:", rf_test_accuracy)

    gb_classifier = HistGradientBoostingClassifier(
        loss='binary_crossentropy',
        learning_rate=0.1,
        max_iter=100,
        max_depth=3,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        max_bins=255,
        random_state=42
    )

    gb_classifier.fit(train_x, train_y)

    gb_train_predictions = gb_classifier.predict(train_x)
    gb_test_predictions = gb_classifier.predict(test_x)

    # Training Accuracy
    gb_train_accuracy = accuracy_score(train_y, gb_train_predictions)
    print("Gradient Boosting Training Accuracy:", gb_train_accuracy)

    # Testing Accuracy
    gb_test_accuracy = accuracy_score(test_y, gb_test_predictions)
    print("Gradient Boosting Testing Accuracy:", gb_test_accuracy)

    print("Random Forest Training Predictions:", rf_train_predictions)
    print("Random Forest Testing Predictions:", rf_test_predictions)
    print("Gradient Boosting Training Predictions:", gb_train_predictions)
    print("Gradient Boosting Testing Predictions:", gb_test_predictions)


if __name__ == '__main__':
    main()
