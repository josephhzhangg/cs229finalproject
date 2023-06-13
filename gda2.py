import util
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import metrics
from gda import GDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import pandas as pd

def main(data, save_path):
    ### LOAD DATASET ###
    train_path = "train_split.csv"
    test_path = "test_split.csv"
    train_x, train_y = util.load_dataset(train_path, add_intercept=False)
    test_x, test_y = util.load_dataset(test_path, add_intercept=False)
    n_features = train_x.shape[1]

    # Beginning of GDA
    gdaModel = LinearDiscriminantAnalysis()
    gdaModel.fit(train_x, train_y)
    train_predictions = gdaModel.predict(train_x)
    test_predictions = gdaModel.predict(test_x)

    # Calculate evaluation metrics for training data
    train_correct = accuracy_score(train_y, train_predictions)
    train_f1 = f1_score(train_y, train_predictions)
    train_recall = recall_score(train_y, train_predictions)
    train_confusion = confusion_matrix(train_y, train_predictions)

    # Calculate evaluation metrics for testing data
    test_correct = accuracy_score(test_y, test_predictions)
    test_f1 = f1_score(test_y, test_predictions)
    test_recall = recall_score(test_y, test_predictions)
    test_confusion = confusion_matrix(test_y, test_predictions)

    # Print evaluation metrics for training data
    print("GDA Training Accuracy:", train_correct)
    print("GDA Training F1 Score:", train_f1)
    print("GDA Training Recall:", train_recall)
    print("GDA Training Confusion Matrix:")
    print(pd.DataFrame(train_confusion, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"]))

    # Print evaluation metrics for testing data
    print("GDA Testing Accuracy:", test_correct)
    print("GDA Testing F1 Score:", test_f1)
    print("GDA Testing Recall:", test_recall)
    print("GDA Testing Confusion Matrix:")
    print(pd.DataFrame(test_confusion, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"]))


if __name__ == '__main__':
    main("breast_cancer_wisconsin_data.csv", "wisconsin_predict.txt")