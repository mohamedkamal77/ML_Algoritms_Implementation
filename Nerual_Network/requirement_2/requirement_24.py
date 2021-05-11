from NN import NN
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris['data']
y = iris['target']
fig, axes = plt.subplots(4,1)
count =0
l_r_l = [0.1,0.01,0.001,0.0001]
acc=[]
for ax in axes.flatten():

    test_acc = []
    train_acc = []
    n = NN(x, y, [3], learning_rate=l_r_l[count], momentum=False)
    n.train()
    ax.plot(np.arange(len(n.test_acc)),n.test_acc,'b')
    ax.plot(np.arange(len(n.test_acc)), n.train_acc, 'r')
    ax.set_title(f"learning_rate = {l_r_l[count]}")
    ax.set(xlabel='Iterations', ylabel='Accuracy')
    acc.append(n.test_acc[-1])
    count += 1
plt.show()
# print(l_r_l,acc)

#the_best = l_r_l[np.argmax(acc)]
the_best  = 0.01

count =1
acc=[]
nodes = []
train_acc=[]
test_acc=[]
yes = True
while yes:

    n = NN(x, y, [count], learning_rate=the_best, momentum=False)
    n.train()
    train_acc.append(n.train_acc)
    test_acc.append(n.test_acc)
    acc.append(n.test_acc[-1])

    if count >1:
        if acc[count-1]<=acc[count-2]  :
            yes = False

    count += 1
    nodes.append(count-1)
print(len(test_acc),count,nodes,acc)
fig, axes = plt.subplots(count-1,1)
count = 0

for ax in  axes.flatten():
    ax.plot(np.arange(len(test_acc[count])),test_acc[count],'b')
    ax.plot(np.arange(len(test_acc[count])), train_acc[count], 'r')
    ax.set_title(f"Nodes = {count+1}")
    ax.set(xlabel='Iterations', ylabel='Accuracy')
    count += 1
plt.show()

