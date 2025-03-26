import torch
from torch import nn

from collections import namedtuple

from confs.definitions import InverseModelConfig
from models.utils import gumbel_softmax

class InverseModel(nn.Module):
	def __init__(self, config: InverseModelConfig, device='cpu'):
		super().__init__()
		self.model_name = 'InverseModel'
		self.config = config
		self.num_actions = config.num_actions
		self.action_embed_dim = config.action_embed_dim
		self.fc1_hidden_dim = config.fc1_hidden_dim
		self.K = config.categorical_dim # number of categories (factors in POMDP terms)
		self.N = config.num_categorical_distributions # dimension for each factor
		self.flat_KN = self.K*self.N
		self.device = device
		
		self.fc1 = nn.Linear((self.flat_KN*2), self.fc1_hidden_dim)
		self.fc2= nn.Linear(self.fc1_hidden_dim, self.fc1_hidden_dim)
		self.fc3 = nn.Linear(self.fc1_hidden_dim , self.action_embed_dim)
		self.act_fn = nn.ELU()

		self.ForwardOuput = namedtuple('ForwardOutput', ['pred_action_emb'])

	def forward(self, state, next_state):
		
		# concat input states
		state_concat = torch.cat([state.squeeze(dim=(-1,-2)), next_state.squeeze(dim=(-1,-2))], dim=1)
		
		h = self.act_fn(self.fc1(state_concat))
		h = self.act_fn(self.fc2(h))
		pred_action_emb = self.fc3(h)

		return self.ForwardOuput(pred_action_emb)