import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from sklearn import linear_model

class LinearRegressionModel :
    
    
    
    
    
    def __init__(self):

        self.theta = []
        self.cost_history=[]
        self.X_train=[]
        self.X_test=[]
        self.Y_train=[]
        self.Y_test=[]
        self.iteration=100


    def compute_cost(self,X, Y, theta):
        """
        Compute cost for linear regression.

        Input Parameters
        ----------------
        X : 2D array where of dimension(m x n)
            m= number of training examples
            n= number of features (including X_0 column of ones)
        Y : array of labels of dimension(1 x m)

        theta : 1D array of weights Oof dimension (1 x n)

        Output Parameters
        -----------------
        J : Scalar value.
        """
        m = len(Y)
        # Perdict Y using theta
        predictions = X.dot(theta)
        
        # get the error 
        errors = np.subtract(predictions, Y)

        # square error  
        sqrErrors = np.square(errors)

        # cost 
        J =  np.sum(sqrErrors)/(2*m)

        return J
    
    def gradient_descent(self,X, Y, theta, alpha, iterations):
        """
        Compute cost for linear regression.

        Input Parameters
        ----------------
        X :  array of dimension(m x n)
            m= number of training examples
            n= number of features (including X_0 column of ones)
        Y : array of labels of dimension(m x 1)
        theta : array of weights of dimension (1 x n)
        alpha : Learning rate.
        iterations: No of iterations. 

        Output Parameters
        -----------------
        theta : array of weights. Dimension (1 x n)
        cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1)
        """
        self.cost_history = np.zeros(iterations)
        m = len(Y)

        for i in range(iterations):

            predictions = X.dot(theta)

            errors = np.subtract(predictions, Y)

            sum_delta = 2*(alpha / m) * X.transpose().dot(errors)

            theta = theta - sum_delta

            self.cost_history[i] = self.compute_cost(X, Y, theta)
        
        return theta

    def perdict(self,X):
        """
        Perdict output For Linear regression

        Input Parameters
        ----------------
         X :  array of dimension(1 x n)
            n= number of features (including X_0 column of ones)

        Output Parameters
        -----------------
        Y_perdicted : perdicted output. Dimension (1 x n)
        cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1)
        """

        Y_perdicted = np.array(X).dot(self.theta)

        return  Y_perdicted


    def train_test_split(self,X,Y):
        """
            Split data into : training data and test data

        """
        m= len(Y)
        m_train = int(m*0.8)
        indeces = np.arange(0,m)
        
        np.random.RandomState(1).shuffle(indeces)
        
        self.X_train , self.X_test = np.array( X[indeces[0: m_train]] ) , np.array( X[indeces[m_train:]] )
        self.Y_train , self.Y_test = np.array( Y[indeces[0: m_train]] ), np.array( Y[indeces[m_train:]] )


    def fit(self,X,Y,alpha=0.1, iterations=100):
        """
            Seprate data to training and test and train the model

        Input Parameters
        ----------------
        X :  array of dimension(m x n)
            m= number of training examples
            n= number of features (excluding X_0 column of ones)
        Y : array of labels of dimension(m x 1)
        alpha : Learning rate.
        iterations: No of iterations. 

        """
        self.theta= np.reshape(np.random.rand(len(X[0]) + 1) , ( len(X[0]) +1,1 ) )
        self.iteration=iterations
        X = np.c_[np.ones((len(X), 1)), X]
        self.train_test_split(X,Y)

        self.theta=self.gradient_descent(self.X_train,self.Y_train,self.theta,alpha,iterations)

    def evaluate_performance(self):
        """
            Evalute model performance over test data
        """
        Y_perdicted = [ self.perdict(i) for i in self.X_test]
        RSS =  np.sum(np.square(np.subtract( Y_perdicted ,self.Y_test  )))
        TSS =  np.sum(np.square(self.Y_test - np.mean(self.Y_test)))

        R_squard = float(1 - (RSS/TSS))
        return f"R_squard error is : {R_squard}"

    def plot_model_over_data(self):
        """
            Plot the data and model

        """
        if (len(self.X_train[0])==2):
            
            plt.scatter( self.X_train[:,1],self.Y_train , color='blue', label= 'Training Data')
            plt.plot(self.X_train[:,1],self.X_train.dot(self.theta), color='green', label='Linear Regression')
            plt.xlabel('Input')
            plt.ylabel('Output')
            plt.title('Linear Regression Fit')
            plt.legend()
            plt.show()
    
        elif (len(self.X_train[0])==3):
           fig,(ax1 , ax2 ,ax3) = plt.subplots(1,3,figsize=(10,6) )
           
           
           ax1.scatter(self.X_train[:,1],self.X_train[:, 2])
           ax1.set_title("input correlation")

           ax2.scatter(self.X_train[:,1],self.Y_train)
           ax2.set_title("Input1 Versus Output")

           ax3.scatter(self.X_train[:,2],self.Y_train)
           ax3.set_title("Input2 Versus Output")
           plt.show()

    def plot_cost_versus_iteration(self):
        """
            Plot the cost function of every iteration

        """
        
        plt.plot(range(1, self.iteration + 1),self.cost_history, color='blue')
    
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost (J)')
        plt.title('Convergence of gradient descent')
        plt.show()
    