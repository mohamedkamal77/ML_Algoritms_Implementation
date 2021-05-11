from treeModel import CART
import numpy as np
import pandas as pd

class RandomForest:
    
    def __init__(self):
        self.tree = CART()

    def train(self,train,max_depth, min_size,n=5):
        train =np.array(train ,dtype=np.int8)
        np.random.RandomState(4).shuffle(train)
        train_data = list(train)
        epoch= int(np.shape(train)[0]/n)
        roots=[]
        for i in range(n):
            if i== n-1:
                temp = train[i*epoch:]
            else:
                temp = train[i*epoch: epoch*(i+1)]
            roots.append(self.tree.build_tree(temp ,max_depth, min_size))
        return roots
    def predict(self,root,row):
        output=[]
        for i in root:
            output.append(self.tree.predict(i,row))
        classes = list(set(output))
        count = 0
        final_output=None
        for j in classes: 
            
            if output.count(j)>count:
                count =output.count(j)
                final_output =j
        
        return j

    def test(self,root , test_data):
        t_data = list(np.array(test_data ,dtype=np.int8))
        result = [ self.predict(root,row) for row in t_data]
        Y_test = np.array(test_data.iloc[:,11])
        accuracy = np.sum(result == Y_test )/len(t_data)
        return f"accurecy : {accuracy}"

