import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """
    Neural Network Model
    """
    def __init__(self, input_size: int, hidden_size: int = 256):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
        self.apply(self._init_weights) # initialize weights
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward propagation
        Args:
            x: input
        Returns: Q values
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.network(x)
    
    def get_action(self, state, epsilon: float = 0.0):
        """
        get action based on our state
        """
        if np.random.random() < epsilon:
            return np.random.randint(4)
            
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values).item()