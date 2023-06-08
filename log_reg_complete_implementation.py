import util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None, k_param = None, sin = False):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta
        self.k_param = k_param
        self.sin = sin

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def loss(self, X, y):
        lossSum = 0
        for i in range(X.shape[0]):
            lossSum += (self.sigmoid( np.dot(X[i], self.theta) ) - y[i])**2
        return lossSum
    
    def gradient(self, X, y):
        return X.T@(self.sigmoid(X@self.theta)-y)
        gradSum = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            gradSum += (y[i] - self.sigmoid(np.dot(X[i], self.theta)))*X[i]
        return gradSum

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        # if (self.sin):
        #     poly = self.create_sin(self.k_param, X)
        # else: 
        #     poly = self.create_poly(self.k_param, X)
        #if (self.sin):
        #    poly = self.create_sin(self.k_param, X)
        #else:
        #    poly = self.create_poly(self.k_param, X)
        #self.theta = np.linalg.solve(np.matmul(poly.T, poly), np.matmul(poly.T, y))

        #self.theta = np.array([-3.43379332, -0.03030307,  1.10850834,  0.4769365,   0.0191778,   0.00375877,
        #    -0.89365538,  0.25550841, -0.0297793,   0.87012357,  0.25611498,  0.40449574,
        #    1.24998454,  0.34385566])

        self.theta = np.zeros(X.shape[1])
        print("starting loss is ", self.loss(X, y))
        l_rate = 0.0000005
        for i in range(10000000):
            self.theta -= l_rate*self.gradient(X, y)
            if i%5000 == 0:
                print("iteration ", i, " loss is: ", self.loss(X,y))
        print("theta is ", self.theta)
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        return np.array([[X[row, 1]**column for column in range(k+1)] for row in range(X.shape[0])])
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        return np.array([[np.sin(X[row, 1]) if column == k+1 else X[row, 1]**column for column in range(k+2)] for row in range(X.shape[0])])
        print("hey")
        print(np.array([[X[row, 1]**column for column in range(k+1)].insert(k+1, np.sin(X[row, 1])) for row in range(X.shape[0])]))
        print(" yooo")
        print(np.array([[X[row, 1]**column for column in range(k+1)] for row in range(X.shape[0])]))

        print(" aaaa")
        print(np.array([[X[row, 1]**column for column in range(k+1)] for row in range(X.shape[0])]))
        print(" beee")
        print(np.array([np.array(X[row, 1]**column for column in range(k+1)).append(np.sin(X[row, 1])) for row in range(X.shape[0])]))
        print(" ceee")
        return np.array([[X[row, 1]**column for column in range(k+1)].append(np.sin(X[row, 1])) for row in range(X.shape[0])])
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        if (self.sin):
            return np.matmul(self.create_sin(self.k_param, X), self.theta)
        else:
            return np.matmul(self.create_poly(self.k_param, X), self.theta)
        # *** END CODE HERE ***

    def performance(self, X, y):
        correct = 0
        truePositive = 0
        falsePositive = 0
        trueNegative = 0
        falseNegative = 0
        wrong = 0
        for i in range(X.shape[0]):
            predict = 1 if self.sigmoid(np.dot(self.theta, X[i]))>=0.5 else 0
            if predict == y[i]:
                correct += 1
                if y[i] == 1:
                    truePositive += 1
                else:
                    trueNegative += 1
            else:
                wrong +=1
                if y[i] == 1:
                    falseNegative += 1
                else:
                    falsePositive += 1

        print("correct: ", correct)
        print("wrong: ", wrong)
        print("ratio correct: ", correct/(correct+wrong), (truePositive+trueNegative)/X.shape[0])
        sensitivity = truePositive/(truePositive+falseNegative)
        specificity = trueNegative/(falsePositive+trueNegative)
        print("sensitivity: ", sensitivity)
        print("specificity: ", specificity)
        print("balanced accuracy: ", (sensitivity + specificity)/2)


def run_exp(train_path, test_path = None, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    test_x,test_y=util.load_dataset(test_path,add_intercept=True)

    # *** START CODE HERE ***
    model = LinearModel()
    #model.fit(train_x, train_y)
    model.theta = [-3.42666653, -0.0263405 ,  0.90874205,  0.59458859,  0.02676357 , 0.00385824,
 -0.60491404,  0.22692407, -0.03771191 , 1.06656541 , 0.2079296 ,  0.27686114,
  1.00010011,  0.32410736]

    print("loss is ", model.loss(train_x, train_y))
    model.performance(train_x, train_y)
    model.performance(test_x, test_y)
    return
    #plt.scatter(train_x[:,1], train_y, label = "train data")
    #plot_y = model.predict(plot_x)
    #plt.scatter(plot_x[:, 1], plot_y, label = "learned hypothesis, k = 3")
    #plt.legend(loc="upper left")
    #plt.show()
    # *** END CODE HERE ***

    # Here plot_y are the predictions of the linear model on the plot_x data
    
    # plt.ylim(-2, 2)
    # plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    # plt.legend()
    # plt.savefig(filename)
    # plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    run_exp(train_path, test_path= eval_path)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train_split.csv',
        small_path='',
        eval_path='test_split.csv')
    
