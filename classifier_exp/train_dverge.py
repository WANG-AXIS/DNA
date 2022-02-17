import torch
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models import CNN, FC
from attacks import PGD_linf
from train import eval
import numpy as np

learning_rate = 5.0e-3
batch_size = 1000
epochs = 10
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MODEL = FC
COR = 'cor'

loss_ce = torch.nn.CrossEntropyLoss()
loss_mse = torch.nn.MSELoss()

def distill_Linf(model, xs, xt, epsilon, k=10, a=2/255):
    device = torch.device("cuda:{}".format(xs.get_device()) if xs.is_cuda else "cpu")
    perturbed_x = xs + (2*epsilon*torch.rand(xs.shape) - epsilon).to(device)
    perturbed_x = torch.clamp(xs, 0, 1)
    model(xt)
    target = model.feat.detach()
    for i in range(k):
        perturbed_x.requires_grad = True
        model.zero_grad()
        model(perturbed_x)
        loss = loss_mse(model.feat, target)
        loss.backward()
        perturbed_x = perturbed_x - a*perturbed_x.grad.data.sign()
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        perturb = torch.clamp(perturbed_x-xs, -epsilon, epsilon)
        perturbed_x = (xs + perturb).detach()
    return perturbed_x


def train(target_model, source_model, data, epsilon, optimizer):
    target_model.train()
    source_model.eval()
    source_loader = iter(DataLoader(data, batch_size=batch_size, shuffle=True))
    target_loader = iter(DataLoader(data, batch_size=batch_size, shuffle=True))
    for _ in range(len(source_loader)):
        xs, ys = next(source_loader)
        xt, yt = next(target_loader)
        xs, ys, xt, yt = xs.to(device), ys.to(device), xt.to(device), yt.to(device)
        x_distilled = distill_Linf(source_model, xs, xt, epsilon, k=20, a=3/255)
        target_model.zero_grad()
        yhat = target_model(x_distilled)
        loss = loss_ce(yhat, ys)
        loss.backward()
        optimizer.step()
        
    
def find_transfer(source_model, target_model, data_loader, attack, epsilon):
    adv_sample_count = 0.0
    adv_transfer_count = 0.0
    for x,y in data_loader:
        x,y = x.to(device), y.to(device)
        x = attack(x,y,source_model,epsilon)
        yhat_s = torch.argmax(source_model(x), dim=1)
        yhat_t = torch.argmax(target_model(x), dim=1)
        adv_s, adv_t = yhat_s!=y, yhat_t!=y
        adv_sample_count = adv_sample_count + torch.sum(adv_s)
        adv_transfer_count = adv_transfer_count + torch.sum(adv_s & adv_t)
    transfer_rate = adv_transfer_count/adv_sample_count 
    return transfer_rate
    
    
def main():
    load_dir = 'saved_models/' + MODEL.__name__ + '/' + COR
    save_dir = 'saved_models_dverge/' + MODEL.__name__ + '/' + COR
    cls1, cls2 = MODEL().to(device), MODEL().to(device)
    cls1.load_state_dict(torch.load(load_dir+'1', map_location=device))
    cls2.load_state_dict(torch.load(load_dir+'2', map_location=device))    
    train_data = MNIST('../mnist_digits/', train=True, download=True,transform=torchvision.transforms.ToTensor())
    test_data = MNIST('../mnist_digits/', train=False, download=True,transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    optimizer1, optimizer2 = optim.Adam(cls1.parameters(), lr=learning_rate), optim.Adam(cls2.parameters(), lr=learning_rate)
    print('###### START TRAINING ######')
    epsilons = [0.05, 0.1, 0.15]
    transfer_rate = np.zeros([len(epsilons), epochs+1])
    for i in range(epochs):
        acc1 = eval(cls1, test_loader)
        acc2 = eval(cls2, test_loader)
        for idx, epsilon in enumerate(epsilons):
            transfer_rate[idx,i] = (find_transfer(cls1, cls2, test_loader, PGD_linf, epsilon)+find_transfer(cls2, cls1, test_loader, PGD_linf, epsilon)).item()/2
        print('Epoch:{} Cls1 Acc:{} Cls2 Acc:{} Transfer Rate:{}'.format(i+1, acc1, acc2, transfer_rate))  
        train(cls1, cls2, train_data, epsilon=0.15, optimizer=optimizer1)
        train(cls2, cls1, train_data, epsilon=0.15, optimizer=optimizer2)

    acc1 = eval(cls1, test_loader)
    acc2 = eval(cls2, test_loader)
    for idx, epsilon in enumerate(epsilons):
        transfer_rate[idx,i+1] = (find_transfer(cls1, cls2, test_loader, PGD_linf, epsilon)+find_transfer(cls2, cls1, test_loader, PGD_linf, epsilon)).item()/2
    print('Epoch:{} Cls1 Acc:{} Cls2 Acc:{} Transfer Rate:{}'.format(i+1, acc1, acc2, transfer_rate))
    print('###### END TRAINING ######')
    torch.save(cls1.state_dict(), save_dir+'1')
    torch.save(cls2.state_dict(), save_dir+'2')
    np.save(save_dir+'trans_rate', transfer_rate)
    
    
                          
        
if __name__ == '__main__':
    main()