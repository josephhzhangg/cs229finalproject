import numpy as np
import util

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        n_rows = x.shape[0]
        n_cols = x.shape[1]

        phi = 0
        mu_0 = np.zeros([1, n_cols])
        mu_1 = np.zeros([1, n_cols])
        sigma = np.zeros([n_cols, n_cols])
        if self.theta == None:
            self.theta = np.zeros([n_cols, 1])
        
        n_pos = 0
        n_neg = 0
        
        for i in range(n_rows):
            if y[i] == 4: # This is changed to 4 for malignant instead of a predicted value of 1
                phi += 1
                mu_1 += x[i]
                n_pos += 1
            else:
                mu_0 += x[i]
                n_neg += 1
        phi = phi / n_rows 
        mu_0 = mu_0 / n_neg
        mu_1 = mu_1 / n_pos

        for i in range(n_rows):
            if y[i] == 1:
                sigma += (x[i] - mu_1).T.dot(x[i] - mu_1)
            else:
                sigma += (x[i] - mu_0).T.dot(x[i] - mu_0)
        sigma = sigma / n_rows 

        # Writing theta in terms of param
        sigmaInv = np.linalg.inv(sigma)
        selfTheta = (-0.5 * (np.matmul(mu_0, sigmaInv.T) - np.matmul(mu_1, sigmaInv.T) + np.matmul(mu_0, sigmaInv) - np.matmul(mu_1, sigmaInv)))
        theta1 = selfTheta[0][0]
        theta2 = selfTheta[0][1]
        theta3 = selfTheta[0][2]
        theta4 = selfTheta[0][3]
        theta5 = selfTheta[0][4]
        theta6 = selfTheta[0][5]
        theta7 = selfTheta[0][6]
        theta8 = selfTheta[0][7]
        theta9 = selfTheta[0][8]


        theta0 = -0.5 * (np.matmul(mu_1, np.matmul(sigmaInv, mu_1.T)) - np.matmul(mu_0, np.matmul(sigmaInv, mu_0.T))) - np.log(1/(phi + 0.0001) - 1)
        self.theta[0] = theta0
        self.theta[1] = theta1
        self.theta[2] = theta2
        self.theta[3] = theta3
        self.theta[4] = theta4
        self.theta[5] = theta5
        self.theta[6] = theta6
        self.theta[7] = theta7
        self.theta[8] = theta8
        self.theta[9] = theta9



    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        print(-x.dot(self.theta))
        p = 1 / (1 + np.exp(-x.dot(self.theta)))


