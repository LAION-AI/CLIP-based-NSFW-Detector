import torch
import torch.nn as nn



class Normalization(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('variance', torch.ones(shape))

    def forward(self, x):
        return (x - self.mean) / (self.variance.sqrt() + 1e-10)
    

class NSFWModel_B32(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Normalization([512])
        self.linear_1 = nn.Linear(512, 32)
        self.linear_2 = nn.Linear(32, 32)
        self.linear_3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.linear_1(x))
        x = self.dropout(self.act(self.linear_2(x)))
        x = self.act_out(self.linear_3(x))
        return x
    
class NSFWModel_L14(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Normalization([768])
        self.linear_1 = nn.Linear(768, 64)
        self.linear_2 = nn.Linear(64, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 1)
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.linear_1(x))
        x = self.act(self.linear_2(x))
        x = self.act(self.linear_3(x))
        x = self.act_out(self.linear_4(x))
        return x
    
class NSFWModel_H14(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)
