import numpy as np
class CART:
    def __init__(self):
        self.root = None
       
    def test_split(self,index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
    
    # Calculate the Gini index for a split dataset
    def gini_index(self,groups, classes):
        # count all samples at split point
        n = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n)
        return gini
    
    # Select the best split point for a dataset
    def get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            uniqu_values = list(set(row[index] for row in dataset))
            for value in uniqu_values:
                groups = self.test_split(index, value, dataset)
                gini = self.gini_index(groups, class_values)
                #print('X%d < %.3f Gini=%.3f' % ((index+1), value, gini))
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, value, gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
 
    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth+1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth+1)
    
    # Build a decision tree
    def build_tree(self, train, max_depth, min_size):
        train =np.array(train ,dtype=np.int8)
        np.random.shuffle(train)
        train_data = list(train)
        root = self.get_split(train_data)
        self.split(root, max_depth, min_size, 1)
        return root
    
    # Print a decision tree
    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))
    
# Make a prediction with a decision tree
    def predict(self,node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    # Make a prediction with a decision tree
    def predict(self,node, row):
    
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']
    def test(self,root , test_data):
        t_data = list(np.array(test_data ,dtype=np.int8))
        result = [ self.predict(root,row) for row in t_data]
        Y_test = np.array(test_data.iloc[:,11])
        accuracy = np.sum(result == Y_test )/len(t_data)
        return f"accurecy : {accuracy}"
