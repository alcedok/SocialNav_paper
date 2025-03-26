import torch
from torch import nn

from collections import namedtuple

from confs.definitions import ActionModelConfig
from models.utils import gumbel_softmax

class ActionModel(nn.Module):
	def __init__(self, config: ActionModelConfig, device='cpu'):
		super().__init__()
		self.model_name = 'ActionModel'
		self.config = config
		self.num_actions = config.num_actions
		self.action_embed_dim = config.action_embed_dim
		self.device = device
		self.ActionEmbedding = nn.Embedding(
			                    num_embeddings=self.num_actions, 
								embedding_dim=self.action_embed_dim)
		self.ForwardOuput = namedtuple('ForwardOutput', ['action_embed'])

	def forward(self, action):
		# embed action
		action_embed = self.ActionEmbedding(action)
		return self.ForwardOuput(action_embed)