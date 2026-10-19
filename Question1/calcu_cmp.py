import math
import random


K=3

"""def dmul(list1,list2):
    result =0.
    for i in range(len(list1)):
            result+=list1[i]*list2[i]
    return result
"""
def mul(list1,list2):
    if isinstance(list1,list):
        if isinstance(list2,list):  #y=Ax   v=Bx
            results = [0 for i in range(len(list1[0]))]
            for i in range(len(results)):
                for j in range(len(results)):
                    results[i] += list1[i][j]*list2[j]
        else:
            results = [0 for i in range(len(list1))]
            for i in range(len(results)):
                results[i] = list1[i] * list2
    else:
        if isinstance(list2,list):   # d_C=d_w[0][1][2]* z  d_B=d_v[0][1][2]* x d_A=d_y[0][1][2]*x
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
                temp+=self.x[j][i]*dout[j]  # dz  
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


if __name__ == '__main__':
    
    #forward
    label = [0 for k in range(K)] 
    x =[random.random() for k in range(K)] #define vector x, dimension K
    A = [[random.random() for i in range(K)] for j in range(K)] #matrices
    layer1 = MulLayer()
    y = layer1.forward(A,x) #list(3)
    layer2 = Sigmoid()
    u = layer2.forward(y) #list(3)
    B = [[random.random() for i in range(K)] for j in range(K)] 
    layer3 = MulLayer()
    v = layer3.forward(B,x) #list(3)
    layer4 = AddLayer()
    z = layer4.forward(u,v) #list(3)
    C = [[random.random() for i in range(K)] for j in range(K)] 
    layer5 = MulLayer()
    w = layer5.forward(C,z)
    L = L2(w)  #calculate loss 
    
    #backward
    d_w = [w[k] / L for k in range(len(w))]
    d_C, d_z = layer5.backward(d_w)
    d_u, d_v = layer4.backward(d_z)
    d_B, _ = layer3.backward(d_v)
    d_y = layer2.backward(d_u)
    d_A, _ = layer1.backward(d_y)



    #Compare
    import torch
    import torch.nn.functional as F
    label_gt = torch.tensor(label)
    x_gt = torch.tensor(x)
    A_gt = torch.tensor(A,requires_grad=True)
    B_gt = torch.tensor(B,requires_grad=True)
    C_gt = torch.tensor(C,requires_grad=True)
    y_gt = torch.matmul(A_gt, x_gt)
    sig = torch.nn.Sigmoid()
    u_gt= sig(y_gt)
    v_gt = torch.matmul(B_gt,x_gt)
    z_gt = u_gt+v_gt
    w_gt = torch.matmul(C_gt,z_gt)
    L_gt = torch.norm(w_gt)
    grad_w = torch.autograd.grad(L_gt, w_gt, retain_graph=True)
    grad_C = torch.autograd.grad(L_gt, C_gt, retain_graph=True)
    grad_z =  torch.autograd.grad(L_gt,z_gt,retain_graph=True)
    grad_u =  torch.autograd.grad(L_gt,u_gt,retain_graph=True)
    grad_v =  torch.autograd.grad(L_gt,v_gt,retain_graph=True)
    grad_y =  torch.autograd.grad(L_gt,y_gt,retain_graph=True)
    grad_B = torch.autograd.grad(L_gt, B_gt, retain_graph=True)
    grad_A =  torch.autograd.grad(L_gt,A_gt,retain_graph=True)
    
    
    


    print(1) # debug over there to check d_A, d_B, d_C and compare with grad_A, grad_B, grad_C
