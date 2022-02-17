import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1,8,3, padding=1)
        self.c2 = nn.Conv2d(8,16,3, padding=1)
        self.c3 = nn.Conv2d(16,32,3, padding=1)
        self.l = nn.Linear(32,10)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(7)
        self.act = nn.ReLU()
        
    def forward(self,x):
        x = self.pool(self.act(self.c1(x)))
        self.feat = self.pool(self.act(self.c2(x)))
        self.maps = self.act(self.c3(self.feat))
        x = self.avgpool(self.maps).flatten(start_dim=1)
        x = self.l(x)
        return x
    
    
class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28**2, 500), nn.ReLU(), nn.BatchNorm1d(500))
        self.l2 = nn.Sequential(nn.Linear(500, 500), nn.ReLU(), nn.BatchNorm1d(500))
        self.l3 = nn.Sequential(nn.Linear(500, 200), nn.ReLU(), nn.BatchNorm1d(200))
        self.l4 = nn.Sequential(nn.Linear(200, 200), nn.ReLU(), nn.BatchNorm1d(200))
        self.l5 = nn.Sequential(nn.Linear(200, 100), nn.ReLU(), nn.BatchNorm1d(100))
        self.l6 = nn.Linear(100,10)
        
    def forward(self,x):
        x = x.flatten(start_dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        self.feat = self.l4(x)
        x = self.l5(self.feat)
        x = self.l6(x)
        return x