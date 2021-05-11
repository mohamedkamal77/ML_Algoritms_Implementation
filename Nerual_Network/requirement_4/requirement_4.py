from NN import NN
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

# iris = datasets.load_iris()
# x = iris['data']
# y = iris['target']
#
# n = NN(x, y, [3], learning_rate=0.01, momentum=True)
# n.train()
# plt.plot(np.arange(len(n.test_acc)),n.test_acc,'b')
# plt.plot(np.arange(len(n.train_acc)),n.train_acc,'r')
# plt.xlabel("iteration")
# plt.ylabel("Accuracy")
# plt.title("2Ires")
# plt.show()
# print("test",n.test_acc[-1],"train",n.train_acc[-1])

df = pd.read_csv("dataset_58_vowel.csv")
all=[]
df=df.drop(columns=["Train_or_Test","Speaker_Number"])
df['Sex'][df['Sex'] == "Male"] = 1
df['Sex'][df['Sex'] == "Female"] = 0
x =df.to_numpy()
n = NN(x[:,0:(x.shape[1]-1)],x[:,(x.shape[1]-1)], [6], learning_rate=0.1, momentum=True)
n.train()
print("test",n.test_acc[-1],"train",n.train_acc[-1])
plt.plot(np.arange(len(n.test_acc)),n.test_acc,'b')
plt.plot(np.arange(len(n.train_acc)),n.train_acc,'r')
plt.xlabel("iteration")
plt.ylabel("Accuracy")
plt.title("Vowel")
plt.show()