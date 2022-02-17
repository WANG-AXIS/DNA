import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models import CNN, FC

lambda1 = 0.05
learning_rate = 5.0e-3
batch_size = 1000
epochs = 20
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MODEL=FC
COR = 'dec'


#Find mutual residual (proxy for mutual information)
def get_R(X,Y):
    X = torch.flatten(X, start_dim=1)
    Y = torch.flatten(Y, start_dim=1)
    #First modify to create nonsingular X:
    _,R = torch.linalg.qr(X)
    cols = torch.diag(R)
    cols = abs(cols/torch.max(cols))>0.0005
    X = X[:,cols]
    X = torch.cat([X, torch.ones([batch_size,1]).to(device)],dim=1)
    Yhat = torch.matmul(torch.matmul(X,torch.linalg.pinv(X)),Y)
    Ehat = Y - Yhat
    SSres = torch.sum(torch.square(Ehat))
    Ybar = torch.mean(Y, dim=0).unsqueeze(0)
    SStot = torch.sum(torch.square(Y-Ybar))
    eta = 0.001
    R = 1 - SSres/(SStot+eta)
    #print('SSres:{} SStot:{} R:{}'.format(SSres, SStot, R))
    return torch.log(SStot+eta)-torch.log(SSres+eta) #R


loss_ce = nn.CrossEntropyLoss()
def get_loss(y,yhat1,yhat2, cls1, cls2):
    L1 = loss_ce(yhat1,y)
    L2 = loss_ce(yhat2,y)
    L3 = get_R(cls1.feat, cls2.feat)
  #  print('L1:{} L2:{} L3:{}'.format(L1, L2, L3))
    return torch.sqrt(L1**2+L2**2)+lambda1*L3

def train(models, data_loader, optimizer):
    models[0].train()
    models[1].train()
    for idx, (x,y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        yhat1 = models[0](x)
        yhat2 = models[1](x)
        for model in models:
            model.zero_grad()
        loss = get_loss(y, yhat1, yhat2, models[0], models[1])
        loss.backward()
        optimizer.step()
        
def eval(model, data_loader):
        model.eval()
        correct_count = 0.0
        total_count = 0.0
        for x,y in data_loader:
          x,y = x.to(device), y.to(device)
          y_hat = model(x)
          correct_count = correct_count + torch.sum(torch.argmax(y_hat, dim=1)==y)
          total_count = total_count + len(y)
        return correct_count/total_count
    
def main():
    cls1, cls2 = MODEL().to(device), MODEL().to(device)
    save_dir = 'saved_models/'+MODEL.__name__+'/'+ COR
    train_data = MNIST('../mnist_digits/', train=True, download=True,transform=torchvision.transforms.ToTensor())
    test_data = MNIST('../mnist_digits/', train=False, download=True,transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(list(cls1.parameters())+list(cls2.parameters()), lr = learning_rate)
    print('###### START TRAINING ######')
    for i in range(epochs):
        train([cls1, cls2], train_loader, optimizer)
        acc1 = eval(cls1, test_loader)
        acc2 = eval(cls2, test_loader)
        print('Epoch:{} Cls1 Acc:{} Cls2 Acc:{}'.format(i+1, acc1, acc2))
    print('###### END TRAINING ######')
    print('Saving: {}'.format(save_dir))
    torch.save(cls1.state_dict(), save_dir+'1')
    torch.save(cls2.state_dict(), save_dir+'2')
                          
        
if __name__ == '__main__':
    main()