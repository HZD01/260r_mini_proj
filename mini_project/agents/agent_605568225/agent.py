import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pathlib

# Set the path to the current folder hosting this file
FOLDER_ROOT = pathlib.Path(__file__).parent

class Actor(nn.Module):
    """
    The actor network for SAC that maps states to a distribution over actions.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mu, log_std
    
    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from the Gaussian distribution
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        
        # Calculate log probability of the action
        log_prob = normal.log_prob(x_t)
        
        # Apply the change of variables formula for the tanh transform
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob


class Critic(nn.Module):
    """
    The critic network for SAC that maps (state, action) pairs to Q-values.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        # Q1 value
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        # Q2 value
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2


class Policy:
    """
    This class is the interface where the evaluation scripts communicate with your trained agent.
    
    You can initialize your model and load weights in the __init__ function. At each environment interaction,
    the batched observation `obs`, a numpy array with shape (Batch Size, Obs Dim), will be passed into the __call__
    function. You need to generate the action, a numpy array with shape (Batch Size, Act Dim=2), and return it.
    
    Do not change the name of this class.
    """
    # FILL YOUR PREFERRED NAME & UID HERE!
    CREATOR_NAME = "Claude"  # Your preferred name here in a string
    CREATOR_UID = "SAC-Agent"  # Your UID here in a string
    
    def __init__(self):
        # MetaDrive observation space is 259-dimensional vector
        self.obs_dim = 259
        # MetaDrive action space is [steering, acceleration]
        self.action_dim = 2
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        
        # Load the trained weights
        self.load_weights()
    
    def load_weights(self):
        """Load pre-trained weights."""
        try:
            model_path = FOLDER_ROOT / "sac_model.pt"
            checkpoint = torch.load(model_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def __call__(self, obs):
        """
        Generate actions from observations.
        
        Args:
            obs: Batch of observations, shape (batch_size, obs_dim)
            
        Returns:
            actions: Batch of actions to take, shape (batch_size, action_dim)
        """
        # Convert numpy array to torch tensor
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        # Set model to evaluation mode
        self.actor.eval()
        
        with torch.no_grad():
            # Sample action from policy
            action, _ = self.actor.sample(obs_tensor)
            
        # Convert action to numpy and ensure correct shape
        action_np = action.cpu().numpy()
        
        return action_np