import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import QNetwork
import random


"""Hyperparameter declaration and values"""

BUFFER_SIZE = int(1e5)    # Memory size of Replay Buffer
BATCH_SIZE = 64           # Minibatch size of sample from reply buffer
GAMMA = 0.9               # Discount on future rewards
LEARN_EVERY = 4           # For learning and performing batch updates from local QNetwork to target QNetwork                  
TAU = 1e-3                # For soft updates
LR = 5e-4                 # Learning rate for Optimizer
eps = 1e-2                # Epsilon , for action to control Exploration-Exploitation Dielemma 

device = torch.device("cuda : 0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with the environment , act and learn from the environment"""

    def __init__(self,state_size,action_size,seed):
        """Initializes Agent object ,
            1. agent variables,
            2. local and target QNetworks,
            3. Optimizer, 
            4. Replay Buffer"""

        self.state_size  = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.t_step = 0

        self.qnetwork_local = QNetwork(state_size,action_size,seed).to(device)
        self.qnetwork_target = QNetwork(state_size,action_size,seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters() , lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def act(self, state, eps= eps ):
        """To read state , pass it through local network and return action values as per the given policy. 
        Then from action values , based on eps gives argmax or chooses a random action from action values"""

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(1,self.action_size))


    def step(self, state, action, reward, next_state, done):
        """Perform a step which consists of,
                1. Add into Replay Buffer
                2. Learn      
        """
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % LEARN_EVERY
        
        if self.t_step == 0 :
            if self.memory.getlength > BATCH_SIZE :
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, GAMMA):
        """Calculate the MSE based on Expected Q value and Target Q value. 
        Use optimizer to learn from MSE and calculate target weights and then update those weights in the target Q network"""

        states, actions, rewards, next_states, done = experiences

        Q_expected = self.qnetwork_local(states).gather(1,actions)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(0)[1].unsqueeze(1)

        Q_targets = rewards + (GAMMA * Q_targets_next * (1-done))

        loss = F.mse_loss(Q_expected , Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Perform inplace copy of target parameters based on Tau"""

        for local_params, target_params in zip(local_model.parameters(), target_model.parameters()):
            target_params.data.copy_(tau*local_params.data + (1-tau)*target_params.data)

class ReplayBuffer():
    """Fixed Size buffer to store experience tuples"""

    def __init__(self,action_size, buffer_size, batch_size, seed):
        """Initializes a ReplayBuffer Object"""

        self.action_size=action_size
        self.memory=deque(maxlen=buffer_size)
        self.batch_size=batch_size
        self.seed=random.seed(seed)

    
    def add(self,state, action, reward, next_state, done):
        """Adds a record in the ReplayBuffer. This method is called by the step method of agent class"""

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Extract BATCH_SIZE number of experience tuples randomly and returns"""
        
        experiences = random.sample(self.memory, k=BATCH_SIZE)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float.to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def getlength(self):
        """Return the current size of ReplayBuffer"""
        return len(self.memory)