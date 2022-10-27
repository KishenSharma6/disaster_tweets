from turtle import forward
import torch
import torch .nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #define layers 
        self,layer1 = nn.Linear(128, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3= nn.Linear(64, 1)
    
    def forward(self, features): #also inherited from nn.Module
        x = self.layer1(features)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
