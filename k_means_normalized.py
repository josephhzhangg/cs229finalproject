from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn import preprocessing
import util
from sklearn.preprocessing import StandardScaler



def init_centroids(num_clusters, data):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    num_features = data.shape[1]
    centroids_init = np.zeros((num_clusters, num_features))
    for i in range(num_clusters):
        centroids_init[i] = data[np.random.randint(0, data.shape[0])]
    #print("centroids: ", centroids_init)
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, data, max_iter=100, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    num_features = data.shape[1]
    num_datapoints = data.shape[0]

    n_clusters = centroids.shape[0]
    newAssign = np.zeros(num_datapoints, dtype = int)
    for it in range(max_iter):
        totalDist = 0
        for i in range(num_datapoints): #in this loop, find the closest cluster to the i-th datapoint
            minDist = float('inf')
            for k in range(n_clusters):
                currDist = np.sum((data[i]-centroids[k])**2)
                if (currDist < minDist):
                    minDist = currDist
                    newAssign[i] = k
            totalDist += minDist
        #print("iteration: ", it, " distance: ", totalDist)
        memberCount = np.zeros(n_clusters)
        newCentroids = np.zeros((n_clusters,num_features))
        for i in range(num_datapoints):
            newCentroids[newAssign[i]] += data[i]
            memberCount[newAssign[i]] += 1

        for k in range (n_clusters):
            newCentroids[k] /= memberCount[k]
        
        if (np.array_equal(centroids, newCentroids)):
            break
        centroids = newCentroids
    return centroids
    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    h = image.shape[0]
    w = image.shape[1]
    n_clusters = centroids.shape[0]
    for i in range(h):
            for j in range(w):
                minDist = float('inf')
                currAssignment = 0
                for k in range(n_clusters):
                    currDist = np.sum((image[i,j,:]-centroids[k])**2)
                    if (currDist < minDist):
                        minDist = currDist
                        currAssignment = k
                image[i,j,:] = centroids[currAssignment]
    # *** END YOUR CODE ***

    return image

def test_centroids(centroids, data, y):
    n_clusters = centroids.shape[0]
    cluster0_pos = 0
    cluster0_neg = 0    
    cluster1_pos = 0
    cluster1_neg = 0
    for i in range(data.shape[0]):
        minDist = float('inf')
        currAssignment = 0
        for k in range(n_clusters):
            currDist = np.sum((data[i]-centroids[k])**2)
            if (currDist < minDist):
                minDist = currDist
                currAssignment = k
        if y[i] == 0:
            if currAssignment == 0:
                cluster0_neg += 1
            else:
                cluster1_neg += 1
        else:
            if currAssignment == 0:
                cluster0_pos += 1
            else:
                cluster1_pos += 1

    print("cluster 0: ", cluster0_pos, cluster0_neg, " , cluster 1: ", cluster1_pos, cluster1_neg)
    cluster0type = 1 if cluster0_pos > cluster0_neg else 0
    cluster1type = 1 if cluster1_pos > cluster1_neg else 0
    if (cluster0type == cluster1type):
        print("BAD: SAME PREFERENCE FOR BOTH CLUSTERS")
        return
    
    correct = 0
    wrong = 0

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(data.shape[0]):
        minDist = float('inf')
        currAssignment = 0
        for k in range(n_clusters):
            currDist = np.sum((data[i]-centroids[k])**2)
            if (currDist < minDist):
                minDist = currDist
                currAssignment = k
        
        if currAssignment == 0:
            if cluster0type == y[i]:
                correct += 1
                if y[i] == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                wrong += 1
                if y[i] == 1:
                    false_negatives += 1
                else:
                    false_positives += 1
        else:
            if cluster1type == y[i]:
                correct += 1
                if y[i] == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                wrong += 1
                if y[i] == 1:
                    false_negatives += 1
                else:
                    false_positives += 1
    

    accuracy = correct / (correct + wrong)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)

    """"
    n_clusters = centroids.shape[0]
    cluster0_pos = 0
    cluster0_neg = 0    
    cluster1_pos = 0
    cluster1_neg = 0
    for i in range(data.shape[0]):
        minDist = float('inf')
        currAssignment = 0
        for k in range(n_clusters):
            currDist = np.sum((data[i]-centroids[k])**2)
            if (currDist < minDist):
                minDist = currDist
                currAssignment = k
        if y[i] == 0:
            if currAssignment == 0:
                cluster0_neg += 1
            else:
                cluster1_neg += 1
        else:
            if currAssignment == 0:
                cluster0_pos += 1
            else:
                cluster1_pos += 1

    print("cluster 0: ", cluster0_pos, cluster0_neg, " , cluster 1: ", cluster1_pos, cluster1_neg)
    cluster0type = 1 if cluster0_pos > cluster0_neg else 0
    cluster1type = 1 if cluster1_pos > cluster1_neg else 0
    if (cluster0type == cluster1type):
        print("BAD: SAME PREFERENCE FOR BOTH CLUSTERS")
        return
    
    correct = 0
    wrong = 0

    for i in range(data.shape[0]):
        minDist = float('inf')
        currAssignment = 0
        for k in range(n_clusters):
            currDist = np.sum((data[i]-centroids[k])**2)
            if (currDist < minDist):
                minDist = currDist
                currAssignment = k
        
        if currAssignment == 0:
            if cluster0type == y[i]:
                correct += 1
            else:
                wrong += 1
        else:
            if cluster1type == y[i]:
                correct += 1
            else:
                wrong += 1
    

    print("correct: ", correct, " wrong: ", wrong, " ratio correct: ", correct/(correct+wrong))
    # sensitivity = truePositive/(truePositive+falseNegative)
    # specificity = trueNegative/(falsePositive+trueNegative)
    # print("sensitivity: ", sensitivity)
    # print("specificity: ", specificity)
    # print("balanced accuracy: ", (sensitivity + specificity)/2)
    """


def main(args):

    # Setup
    #np.random.seed(0)
    num_clusters = 2
    train_path='train_split.csv'
    test_path='test_split.csv'
    train_x,train_y=util.load_dataset(train_path,add_intercept=False)
    test_x,test_y=util.load_dataset(test_path,add_intercept=False)

    #normalize the data
    scaler1 = StandardScaler()
    scaler1.fit(train_x)
    train_x = scaler1.transform(train_x)
    scaler2 = StandardScaler()
    scaler2.fit(test_x)    
    test_x = scaler2.transform(test_x)

    # Initialize centroids
    #print('[INFO] Centroids initialized')
    for i in range(100):
        np.random.seed(i)
        centroids_init = init_centroids(num_clusters, train_x)

        # Update centroids
        #print(25 * '=')
        #print('Updating centroids ...')
        #print(25 * '=')
        centroids = update_centroids(centroids_init, train_x, max_iter=1000)

        # Test centroid assignemnts
        #print("testing test data")
        #test_centroids(centroids, train_x, train_y)
        # Test centroid assignments on training data
        print("Testing training data")
        test_centroids(centroids, train_x, train_y)

        # Test centroid assignments on testing data
        print("Testing testing data")
        test_centroids(centroids, test_x, test_y)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
