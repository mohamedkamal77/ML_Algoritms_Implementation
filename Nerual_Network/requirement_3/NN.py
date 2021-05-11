import numpy as np
from sklearn.model_selection import train_test_split
class NN:
    def __init__(self, x, y, dimensions_list,learning_rate,momentum=False,test=0.3):
        self.momentum = momentum
        self.prev_grad = 0
        self.learning_rate = learning_rate
        self.y = y
        self.x = x
        self.parameters = {}
        self.cost = []
        self.train_acc = []
        self.test_acc = []
        output_classes = list(set(y))
        self.d_list = [np.shape(x)[1]] + dimensions_list + [len(output_classes)]
        print(self.d_list)
        self.initialize_parameters()
        self.hot_encoding(output_classes)
        self.x_train,  self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y,test_size=test,random_state=0)
        print(np.shape(self.x_train),np.shape( self.y_train), np.shape(self.x_test), np.shape(self.y_test))
        self.y_train= np.array(self.y_train)
        self.y_test= np.array(self.y_test)
        self.x_train = np.array(self.x_train,dtype=np.float)
        self.x_test = np.array(self.x_test,dtype=np.float)

        #self.a.insert(0,np.copy(self.x))


    def initialize_parameters(self):
        L = len(self.d_list)  # number of layers in the network

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.RandomState(133).randn(self.d_list[l], self.d_list[l - 1])*0.01
            self.parameters['b' + str(l)] = np.zeros((self.d_list[l], 1))


    def hot_encoding(self, output_classes):
        y = [[]]* np.size(self.y)
        for i, output in enumerate(output_classes):
            temp = [0]*len(output_classes)
            temp[i] = 1
            indeces = np.where(np.asarray(self.y) == output)[0]
            for index in indeces:
                y[index] = temp
        self.y = y

    def sigmoid(self, z):


        a = 1 / (1 + np.exp(-1*z.astype(np.float)))
        cache = z

        return a, cache


    def sigmoid_backward(self, da, cache):

        z = cache

        s = 1 / (1 + np.exp(-z.astype(np.float)))
        dz = da * s * (1 - s)

        #assert (dZ.shape == Z.shape)

        return dz

    def tanh(self,z):
        return np.tanh(z),z

    def tanh_backward(self,da,cach):
        z = cach
        a = np.tanh(z)

        return da*(1 - a*a)

    def linear_forward(self, a, w, b):
        z = a@w.T+ b.T
        cache = (a, w, b)
        return z, cache

    def linear_activation_forward(self, a_prev, w, b, activation):

        if activation == "sigmoid":

            z, linear_cache = self.linear_forward(a_prev, w, b)
            a, activation_cache = self.sigmoid(z)

        elif activation == "relu":
            #
            z, linear_cache = self.linear_forward(a_prev, w, b)
            a, activation_cache = self.tanh(z)

        cache = (linear_cache, activation_cache)

        return a, cache

    def L_model_forward(self, x):
        caches = []
        a = x
        L = len(self.parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            a_prev = a
            a, cache = self.linear_activation_forward(
                 a_prev, self.parameters['W{:d}'.format(l)],
                 self.parameters['b{:d}'.format(l)],
                 activation='relu')
            caches.append(cache)

        al, cache = self.linear_activation_forward(a, self.parameters['W%d' % L],
                                                   self.parameters['b%d' % L],
                                                   activation='sigmoid')
        caches.append(cache)

        #assert (al.shape == (1, x.shape[1]))
        return al, caches

    def compute_cost(self, al, y):



        cost = -1 / np.sum(y * np.log(al+0.00001) + (1-y) * np.log(1-al+0.00001))#- np.sum(y*np.log(al + 0.00001) )
        m = y.shape[1]
        # print("m",m)
        return -np.sum((y-al)*(y-al))/m#- np.sum(y*np.log(al + 0.0000000001) )#np.sum(y*np.log(al + 0.0000000001) )

    def linear_backward(self, dz, cache):

        a_prev, w, b = cache
        m = a_prev.shape[1]
        dw = 1 / m * dz.T @ a_prev
        db = 1 / m * np.sum(dz, axis=0, keepdims=True)
        db = db.T
        da_prev = dz@w


        return da_prev, dw, db

    def linear_activation_backward(self, da, cache, activation):

        linear_cache, activation_cache = cache

        if activation == "relu":
            dz = self.tanh_backward(da, activation_cache)
            da_prev, dw, db = self.linear_backward(dz, linear_cache)

        elif activation == "sigmoid":
            dz = self.sigmoid_backward(da,activation_cache)
            da_prev, dw, db = self.linear_backward(dz, linear_cache)

        return da_prev, dw, db

    def L_model_backward(self, al, y, caches):
        grads = {}
        L = len(caches)  # the number of layers
        #m = al.shape[1]
        y = y.reshape(al.shape)  # after this line, Y is the same shape as AL


        #dal = - (np.divide(y, al) - np.divide(1 - y, 1 - al))#2*(y-al)
        #ind = np.argmax(y)
        # zz = np.zeros(dal.shape)
        # zz[ind] = 1
        #print(ind, dal.shape)
        #dal[ind]= dal[ind]*1.5

        dal =-2*(y-al)/y.shape[1]# np.divide(y,al+0.0000000001)#- 2*(y-al)#- np.divide(1 - y, 1 - al) - (y-al) *np.divide(y,al) #/ np.abs(y-al) / np.divide(y,al) # - (np.divide(y, al)
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] =\
            self.linear_activation_backward(dal, current_cache, 'sigmoid')


        for l in reversed(range(L - 1)):

            current_cache = caches[l]
            da_prev_temp, dw_temp, db_temp = \
                self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
            grads["dA" + str(l)] = da_prev_temp
            grads["dW" + str(l + 1)] = dw_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads):


        L = len(self.parameters) // 2  # number of layers in the neural network

        if self.prev_grad == 0 or self.momentum==False:
            for l in range(L):
                self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - self.learning_rate * grads["dW" + str(l + 1)]
                self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - self.learning_rate * grads["db" + str(l + 1)]
        else:
            for l in range(L):
                self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - self.learning_rate * grads["dW" + str(l + 1)] -0.05*self.prev_grad["dW" + str(l + 1)]
                self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - self.learning_rate * grads["db" + str(l + 1)] -0.05*self.prev_grad["db" + str(l + 1)]
            self.prev_grad = np.copy(grads)


    def train(self):

        for i in range(10000):
            np.random.RandomState(i).shuffle(self.x_train)
            np.random.RandomState(i).shuffle(self.y_train)

            al, caches = self.L_model_forward(self.x_train)
            cost = self.compute_cost(al, self.y_train)

            self.cost.append(cost)
            if i%10==0 and i>20:
                if abs(np.mean(self.cost[i-10:i]) - np.mean(self.cost[i-20:i-10]))<0.0001:
                    break
            grads = self.L_model_backward(al,self.y_train,caches)
            self.update_parameters(grads)
            print(cost)
            if i%10 == 0:
                train = self.test(self.x_train,self.y_train)
                test = self.test(self.x_test,self.y_test)
                self.train_acc.append(train)
                self.test_acc.append(test)



    def test(self,x_,y_):

        al, _ = self.L_model_forward(x_)
        return np.sum(np.argmax(al,axis=1)==np.argmax(y_,axis=1))/y_.shape[0]

