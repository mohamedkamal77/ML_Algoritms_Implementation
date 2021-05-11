import numpy as np
from LinearRegressionModel import LinearRegressionModel

# load the data

data = np.genfromtxt('data/univariateData.dat',
                     delimiter=',')

X = data[:,0]
Y = data[:,1]
X =np.reshape(X ,(len(Y),1))
Y =np.reshape(Y ,(len(Y),1))

model = LinearRegressionModel()
model.fit(X,Y,0.003,3000)
performance =model.evaluate_performance()
print(performance)
model.plot_cost_versus_iteration()
model.plot_model_over_data()
