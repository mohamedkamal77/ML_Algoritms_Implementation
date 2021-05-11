import numpy as np
from LinearRegressionModel import LinearRegressionModel

# load the data
data = np.genfromtxt('data/multivariateData.dat',
                     delimiter=',')

X = data[:,0:2]
Y = data[:,2]
X =np.reshape(X ,(len(Y),2))
Y =np.reshape(Y ,(len(Y),1))

model = LinearRegressionModel()
model.fit(X,Y,10**-11*1.7,200000)
performance =model.evaluate_performance()
print(performance)
model.plot_cost_versus_iteration()
model.plot_model_over_data()