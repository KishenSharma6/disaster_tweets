import torch
import torch.nn as nn

#Set seed
torch.manual_seed(6)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()

        self.rnn = torch.nn.RNN(input_size, 
                                hidden_size, 
                                nonlinearity = 'relu',
                                batch_first = True, 
                                )
        self.linear = torch.nn.Linear(hidden_size, 
                                      output_size)
    
    def forward(self, x):
        h = self.rnn(x)[0]
        x = self.linear(h) 
        return x

