import util
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import metrics
from gda import GDA

# Note in data.txt, attributes 2-10 are on a scale 1-10 and class attribute at the end is 2 for benign and 4 for malignant, missing values were changed to 11

"""
The overall class distribution is 458 benign (66%) to 241 malignant (34%)
"""


def main(data, save_path):
  util.split_dataset(data)
  x_train, y_train = util.load_dataset(data + "_train")
  x_test, y_test = util.load_dataset(data + "_test")
  #print(x_train, y_train, x_test, y_test)

  linReg = LinearRegression() # To be replaced with our own linear regression functionality
  linReg.fit(x_train, y_train)
  #plot_path = save_path.replace('.csv', '.png')
  predictions = linReg.predict(x_test)
  #print(predictions)
  classified_predictions = util.classifyPredictions(predictions, 2, 4) # Where 2 and 4 are the label predictions for y
  #print(classified_predictions)
  #print(y_test)
  
  #util.plot(x_test, y_test, linReg.theta, plot_path)
  np.savetxt(save_path, predictions)

  # Finding the accuracy
  correct = 0
  for i, pred in enumerate(classified_predictions):
     if pred == y_test[i]:
        correct += 1
  accuracy = correct / len(y_test)
  print("linear regression accuracy: " + str(accuracy))


  # Beginning of GDA
  gdaModel = GDA()
  gdaModel.fit(x_train, y_train)
  predictions2 = gdaModel.predict(x_test)
  classified_predictions2 = util.classifyPredictions(predictions, 2, 4) # Where 2 and 4 are the label predictions for y
  correct = 0
  for i, pred in enumerate(classified_predictions):
     if pred == y_test[i]:
        correct += 1
  accuracy2 = correct / len(y_test)
  print("GDA accuracy: " + str(accuracy2))

  # Beginning of SVM
  svmModel = svm.SVC(kernel = "linear")
  print("fitting SVM model")
  svmModel.fit(x_train, y_train) # I don't know why my code seems to stall here
  print("model fitted")
  predictions3 = svmModel.predict(x_test) 
  print("svm predictions made") 
  accuracy3 = metrics.accuracy_score(y_test, predictions3)
  print("svm accuracy: " + str(accuracy3))


if __name__ == '__main__':
    main("breast_cancer_wisconsin_data.csv", "wisconsin_predict.txt")
