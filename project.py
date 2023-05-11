import util

# Note in data.txt, attributes 2-10 are on a scale 1-10 and class attribute at the end is 2 for benign and 4 for malignant, missing values were changed to 11

"""
The overall class distribution is 458 benign (66%) to 241 malignant (34%)
"""


def main():

  util.split_dataset("breast_cancer_wisconsin_data.csv")
  x_train, y_train = util.load_dataset("breast_cancer_wisconsin_data.csv_train")
  x_test, y_test = util.load_dataset("breast_cancer_wisconsin_data.csv_test")
  print(x_train, y_train, x_test, y_test)




if __name__ == '__main__':
    main()
