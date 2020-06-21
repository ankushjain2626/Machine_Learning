import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Policy Model which the actor will use"""

    def __init__(self,state_size,action_size,seed,fc1_nodes=64,fc2_nodes=64):
        """
        Initializes parameters and build model

        Params
        ==========
            state_size (int) : Dimension of input(each state)
            action_size(int) : Dimension of output(each action)
            seed(int)        : Random seed
            fc1_nodes(int)   : Number of nodes in first fully connected layer(first hidden layer)
            fc2_nodes(int)   : Number of nodes in second fully connected layer(second hidden layer)
        """
        super.__init__(QNetwork,self)
        self.fc1 = nn.Linear(state_size,fc1_nodes)
        self.fc2 = nn.Linear(fc1_nodes,fc2_nodes)
        self.fc3 = nn.Linear(fc2_nodes,action_size)
        self.seed = torch.manual_seed(seed)

    def forward(self,state):
        """Pass a state(input) through the network and return the output(action). In other words,it maps, states --> actions
        
        Params
        ==========
            state(int)  : State vector of size state_size
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))
