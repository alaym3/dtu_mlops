from torch import nn
from torch.nn.modules.conv import Conv2d

from torch import unsqueeze


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # build convolutional layers
        self.conv = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, padding=1, stride =1),
                                nn.Dropout(p=0.2),
                                nn.ReLU(),
                                nn.MaxPool2d(2,2),
                                nn.Conv2d(4,8,3,1,1),
                                nn.Dropout(p=0.2),
                                nn.ReLU(),
                                nn.Conv2d(8,8,3,1,1),                                 
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.ReLU(),
                                nn.MaxPool2d(2,2), 
                                nn.Conv2d(8,16,3,1,1),
                                nn.Dropout(p=0.2),
                                nn.ReLU())
        
        # build fully connected layers
        self.fc = nn.Sequential(nn.Linear(7*7*16, 256),
                                nn.Dropout(p=0.2),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.Dropout(p=0.2),
                                nn.ReLU(),
                                nn.Linear(128,10),
                                nn.LogSoftmax(dim=1))
        
    # forward pass    
    def forward(self, x):
        x = unsqueeze(x,dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out  
    
