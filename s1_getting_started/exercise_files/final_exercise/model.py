from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(784, 128)
        # Inputs to 2nd hidden layer linear transformation
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        # Hidden layer 1 with relu activation
        x = F.relu(self.fc1(x))
        # Hidden layer 2 with relu activation
        x = F.relu(self.fc2(x))
        # Output layer with softmax activation
        x = F.log_softmax(self.output(x), dim=1)
        
        return x