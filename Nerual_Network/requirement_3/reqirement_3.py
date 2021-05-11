import numpy as np
import pandas as pd
from NN import NN
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_58_vowel.csv")
all=[]
df=df.drop(columns=["Train_or_Test","Speaker_Number"])
df['Sex'][df['Sex'] == "Male"] = 1
df['Sex'][df['Sex'] == "Female"] = 0
x =df.to_numpy()
# fig, axes = plt.subplots(5,1)
# count =0
# l_r_l = [0.1,0.01,0.001,0.0001,0.00001]
# acc=[]
# for ax in axes.flatten():
#
#     n = NN(x[:,0:(x.shape[1]-1)],x[:,(x.shape[1]-1)],[4],l_r_l[count],False,test=0.25)
#     n.train()
#     test = ax.plot(np.arange(len(n.test_acc)),n.test_acc,'b')
#     train = ax.plot(np.arange(len(n.test_acc)), n.train_acc, 'r')
#
#     ax.set_title(f"learning_rate = {l_r_l[count]}")
#     ax.set(xlabel='Iterations', ylabel='Accuracy')
#     acc.append([n.test_acc[-1],n.train_acc[-1]])
#     count += 1
# plt.show()
# print(l_r_l,acc)
# all.append([l_r_l,acc])
# acc = np.array(acc)
# the_best = l_r_l[np.argmax(acc[:,1])]
the_best = 0.1
# count =0
# acc=[]
# nodes = []
# train_acc=[]
# test_acc=[]
# yes = True
# while yes:
#
#     n = NN(x[:,0:(x.shape[1]-1)],x[:,(x.shape[1]-1)], [count+1], learning_rate=the_best, momentum=False)
#     n.train()
#     train_acc.append(n.train_acc)
#     test_acc.append(n.test_acc)
#
#     acc.append([n.test_acc[-1],n.train_acc[-1]])
#     if count >1:
#         if acc[count][1]<=acc[count-1][1] :
#             yes = False
#
#     count += 1
#     nodes.append(count)
# print(count,nodes,acc)
# all.append([nodes,acc])
# fig, axes = plt.subplots(count,1)
# count = 0
#
# for ax in  axes.flatten():
#     ax.plot(np.arange(len(test_acc[count])),test_acc[count],'b')
#     ax.plot(np.arange(len(test_acc[count])), train_acc[count], 'r')
#     ax.set_title(f"Nodes = {count+1}")
#     ax.set(xlabel='Iterations', ylabel='Accuracy')
#     count += 1
# plt.show()
n = NN(x[:,0:(x.shape[1]-1)],x[:,(x.shape[1]-1)], [6,4], learning_rate=the_best, momentum=False)
n.train()
print(all)

print("test",n.test_acc[-1],"train",n.train_acc[-1],the_best)

plt.plot(np.arange(len(n.test_acc)),n.test_acc,'b')
plt.plot(np.arange(len(n.train_acc)),n.train_acc,'r')
plt.xlabel("iteration")
plt.ylabel("Accuracy")
plt.title("2_layer")
plt.show()