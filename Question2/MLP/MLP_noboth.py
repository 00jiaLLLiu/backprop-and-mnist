import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])  

train_dataset = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)


    def forward(self, x):
        x = x.view(-1, 784)     # .view() -1 calculate automatically
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        out = self.l5(x)        # dont need relu
        return out

model = MLP().to(device)

criterion = torch.nn.CrossEntropyLoss()      
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

if __name__ == '__main__':
    Epoch = 10
    type, step_list, epoch_list, train_loss, train_acc = [], [], [], [], []
    step=1

    start_time = time.time()
    
    for epoch in range(Epoch):
        epoch_start = time.time()
        runnning_loss_step, running_loss_epoch = 0.0, 0.0
        correct_step, correct_epoch = 0, 0
        total_step, total_epoch= 0, 0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            inputs,target = inputs.to(device),target.to(device)
            optimizer.zero_grad()
            # forward, backward, update
            outputs = model(inputs)
            loss = criterion(outputs, target)
            _, predicted = torch.max(outputs.data, dim=1)
            
            #save total number to calculate accuracy
            total_epoch += target.size(0)   #total number of every epoch
            total_step += target.size(0)    #total number of every step
             
            #save correct number to calculate accuracy
            correct_epoch += (predicted == target).sum().item()  #correct number of every epoch 
            correct_step += (predicted == target).sum().item()    #accuracy of every step
            
            loss.backward()
            optimizer.step()
            
            running_loss_epoch += loss.item()
            runnning_loss_step += loss.item()
            
            if batch_idx % 300 == 299:
                #print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, runnning_loss_step / 300))
                type.append('mlp_noBD')     
                step_list.append(step)
                epoch_list.append(epoch+1)
                train_loss.append(runnning_loss_step / 300) 
                train_acc.append(correct_step / total_step)
                #print("train_acc",correct_step/total_step)
                step+=1
                correct_step, runnning_loss_step, total_step = 0.0, 0.0, 0
        epoch_end = time.time() 
        
        print('Epoch: {},Time :{}s ,Loss : {} ,Accuracy : {}'.format(epoch+1,(epoch_end-epoch_start),(running_loss_epoch/total_epoch),(correct_epoch/total_epoch)))
    result = pd.DataFrame(list(zip(type,step_list,epoch_list,train_loss,train_acc)), columns=['type','step','epoch','train_loss','train_acc'])   
    result.to_csv("MLP.csv",mode='a',header=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    print('accuracy on test set: %.2f %% ' % (100 * correct / total))
    print('Total Time is {}s'.format(end_time-start_time))

