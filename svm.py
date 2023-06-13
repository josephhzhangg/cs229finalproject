import util 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

def main():
    ### LOAD DATASET ###
    train_path = "train_split.csv"
    test_path = "test_split.csv"
    train_x, train_y = util.load_dataset(train_path, add_intercept=False)
    test_x, test_y = util.load_dataset(test_path, add_intercept=False)
    n_features = train_x.shape[1]

    # Beginning of SVM with RBF kernel
    svmModel = SVC(kernel='rbf')  # Also utilized rbf, poly, linear kernels
    svmModel.fit(train_x, train_y)
    train_predictions = svmModel.predict(train_x)
    test_predictions = svmModel.predict(test_x)

    # Calculate evaluation metrics for training data
    train_accuracy = accuracy_score(train_y, train_predictions)
    train_f1 = f1_score(train_y, train_predictions)
    train_recall = recall_score(train_y, train_predictions)
    train_confusion = confusion_matrix(train_y, train_predictions)

    # Calculate evaluation metrics for testing data
    test_accuracy = accuracy_score(test_y, test_predictions)
    test_f1 = f1_score(test_y, test_predictions)
    test_recall = recall_score(test_y, test_predictions)
    test_confusion = confusion_matrix(test_y, test_predictions)

    # Print evaluation metrics for training data
    print("SVM Training Accuracy:", train_accuracy)
    print("SVM Training F1 Score:", train_f1)
    print("SVM Training Recall:", train_recall)
    print("SVM Training Confusion Matrix:")
    print(train_confusion)

    # Print evaluation metrics for testing data
    print("SVM Testing Accuracy:", test_accuracy)
    print("SVM Testing F1 Score:", test_f1)
    print("SVM Testing Recall:", test_recall)
    print("SVM Testing Confusion Matrix:")
    print(test_confusion)


if __name__ == '__main__':
    main()