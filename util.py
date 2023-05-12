import matplotlib.pyplot as plt
import numpy as np
import csv
import random

def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    print("Loading dataset")
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    print(headers)
    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i] != label_col]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels

def split_dataset(filename, train_ratio=0.9, seed=42):
    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Extract the header
    header = lines[0]
    
    # Remove the header from the lines
    lines = lines[1:]
    
    # Shuffle the lines randomly
    random.seed(seed)
    random.shuffle(lines)
    
    # Calculate the split index based on the train_ratio
    split_index = int(len(lines) * train_ratio)
    
    # Split the lines into training and test sets
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]
    
    # Prepend the header to each set
    train_lines = [header] + train_lines
    test_lines = [header] + test_lines

    with open(filename + "_train", 'w') as train_file:
      train_file.writelines(train_lines)

    with open(filename + "_test", 'w') as test_file:
      test_file.writelines(test_lines)
    
    return train_lines, test_lines


def classifyPredictions(predictions, class1, class2):
    ret = []
    for pred in predictions:
        if abs(class1 - pred) > abs(class2 - pred):
            ret.append(float(class2))
        else:
            ret.append(float(class1))
    
    return ret