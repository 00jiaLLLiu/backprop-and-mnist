import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

K = 3 #dimension K   only one x

"""
def dmul(list1,list2):
    result =0.
    for i in range(len(list1)):
            result+=list1[i]*list2[i]
    return result"""

def mul(list1,list2):
    if isinstance(list1,list):
        if isinstance(list2,list):
            results = [0 for i in range(len(list1[0]))]
            for i in range(len(results)):
                for j in range(len(results)):
                    results[i] += list1[i][j]*list2[j]
        else:
            results = [0 for i in range(len(list1))]
            for i in range(len(results)):
                results[i] = list1[i] * list2
    else:
        if isinstance(list2,list):
            results = [0 for i in range(len(list2))]
            for i in range(len(results)):
                results[i] = list1 * list2[i]
        else:
            results = list1*list2

    return results

class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        results = [0 for i in range(len(x))]
        for i in range(len(results)):
            results[i] = 1/(1+math.exp(-x[i]))
        self.out = results

        return results
    def backward(self,dout):
        results = [0 for i in range(len(dout))]
        for i in range(len(results)):
            results[i] =dout[i]*(1-self.out[i])*self.out[i]
        return results

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = mul(self.x,self.y)
        return out
    def backward(self,dout):
        results_x = []
        retults_y = []
        for i in range(len(self.y)):
            results_x.append(mul(dout[i],self.y))
            temp = 0.
            for j in range(len(dout)):
                temp+=self.x[j][i]*dout[j]
            retults_y.append(temp)
        return results_x,retults_y

class AddLayer:
    def __init__(self):
        pass
    def forward(self,x,y):
        results = [0 for i in range(len(x))]
        for i in range(len(results)):
            results[i] = x[i]+y[i]

        return results
    def backward(self,dout):
        dx = dout*1
        dy = dout*1
        return dx,dy

def L2(pred):
    result = 0.
    for i in range(len(pred)):
        result+=math.pow(pred[i],2)
    return math.sqrt(result)

def updata(matrix,grad_matrix,learning_rate):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] -= grad_matrix[i][j]*learning_rate
    return matrix

if __name__ == '__main__':
    Epoch = 100
    learning_rate = 0.01 #
    label = [0 for k in range(K)]  
    x = [random.random() for k in range(K)]  
    A = [[random.random() for i in range(K)] for j in range(K)]  
    B = [[random.random() for i in range(K)] for j in range(K)]  
    C = [[random.random() for i in range(K)] for j in range(K)]  
    layer1 = MulLayer()
    layer2 = Sigmoid()
    layer3 = MulLayer()
    layer4 = AddLayer()
    layer5 = MulLayer()
    Loss_xi = []            # the loss of x1/x2/x3 in every epoch
    for epoch in range(Epoch):
        
        #forward
        y = layer1.forward(A,x)
        u = layer2.forward(y)
        v = layer3.forward(B,x)
        z = layer4.forward(u,v)
        w = layer5.forward(C,z)
        L = L2(w)  
        Loss_xi.append(L) 
        
        #backward
        dL = [w[k]/L for k in range(len(w))]
        d_C,d_z = layer5.backward(dL)
        C = updata(C,d_C,learning_rate)
        d_u,d_v = layer4.backward(d_z)
        d_B, _ = layer3.backward(d_v)
        B = updata(B, d_B, learning_rate)
        d_y = layer2.backward(d_u)
        d_A,_= layer1.backward(d_y)
        A = updata(A, d_A, learning_rate)
        
    pltx = [i for i in range(Epoch)]  
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(pltx,Loss_xi)
    plt.show()