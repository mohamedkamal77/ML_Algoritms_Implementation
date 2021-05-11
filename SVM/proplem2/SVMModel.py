from sklearn import datasets
import numpy as np
import Utilities
import matplotlib.pyplot as plt


class SVM:

    def __init__(self, C=10000, learning_rate=0.00001, iterations=10000):
        """

        :type learning_rate: object
        """
        self.C = C
        self.learning_rate = learning_rate
        self.iterations = iterations
        iris = datasets.load_iris()
        utilities = Utilities.Utilities()
        self.X = utilities.scaling_features(iris["data"][:, (2, 3)])
        ones = np.ones((self.X.shape[0], 1))  # petal length, petal width
        self.X = np.column_stack((self.X, ones))  # add intercept column
        self.Y = utilities.scaling_labels((iris["target"] == 2).astype(np.float64))
        self.X_train, self.X_test, self.Y_train, self.Y_test = utilities.tt_split(self.X, self.Y)
        # print(self.Y_train,self.Y_test)
        self.weights = np.zeros(self.X.shape[1])


    def compute_cost(self, W, X, Y):
        # calculate loss
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        loss = self.C * (np.sum(distances) / N)

        # calculate cost
        cost = 1 / 2 * np.dot(W, W) + loss
        return cost

    def calculate_cost_gradient(self, W, X, Y):
        # if only one example is passed (eg. in case of SGD)
        distance = 1 - (Y * np.dot(X, W))
        dw = np.zeros(len(W))
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.C * Y[ind] * X[ind])
            dw += di
        dw = dw / len(Y)  # average
        return dw

    def fit(self):

        for i in range(self.iterations):
            # for index, x in enumerate(self.X_train):
            dw = self.calculate_cost_gradient(self.weights, self.X_train, self.Y_train)
            self.weights = self.weights - (self.learning_rate * dw)

    def predict(self, X):
        return np.sign(np.dot(X, self.weights))

    def test(self):
        y_pred = np.sign(np.dot(self.X_test, self.weights))
        print(y_pred)
        accuracy = np.sum(y_pred == self.Y_test) / np.size(self.Y_test)
        print("accurecy is: ", accuracy)

    def draw(self, X, Y, text):
        from matplotlib.colors import ListedColormap
        X_set, y_set = X[:, 0: (np.shape(X)[1] - 1)], Y
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 0.1, stop=X_set[:, 0].max() + 0.1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 0.1, stop=X_set[:, 1].max() + 0.1, step=0.01))

        x = np.column_stack((np.array(X1.ravel()).T, np.array(X2.ravel()).T, np.ones(np.shape(X2.ravel()))))
        plt.contourf(X1, X2, self.predict(x).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('white', 'black'))(i), label=j)
        plt.title(f'SVM ({text} set)')
        plt.xlabel('petal length')
        plt.ylabel('petal width')
        plt.legend()
        plt.show()



