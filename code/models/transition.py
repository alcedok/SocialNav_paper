import torch
from torch import nn

from collections import namedtuple

from confs.definitions import TransitionModelConfig
from models.utils import gumbel_softmax

class TransitionModel(nn.Module):
	def __init__(self, config: TransitionModelConfig, device='cpu'):
		super().__init__()
		self.model_name = 'TransitionModel'
		self.config = config
		self.num_actions = config.num_actions
		self.init_gumbel_temperature = config.gumbel_temperature
		self.action_embed_dim = config.action_embed_dim
		self.fc1_hidden_dim = config.fc1_hidden_dim
		self.K = config.categorical_dim# number of categories (factors in POMDP terms)
		self.N = config.num_categorical_distributions # dimension for each factor
		self.flat_KN = self.K*self.N
		self.use_belief_action_cross_attn = config.use_belief_action_cross_attn
		self.device = device
		
		self.fc1 = nn.Linear(self.flat_KN+self.action_embed_dim, self.fc1_hidden_dim)
		self.fc2= nn.Linear(self.fc1_hidden_dim, self.fc1_hidden_dim)
		self.fc3 = nn.Linear(self.fc1_hidden_dim , self.flat_KN)
		self.act_fn = nn.ELU()

		# attention		
		self.attention_belief = nn.MultiheadAttention(embed_dim=self.K, num_heads=self.K, batch_first=True)

		if self.use_belief_action_cross_attn:
			self.attention_cross_belief_action = nn.MultiheadAttention(
									embed_dim=self.K, 
									num_heads=self.K,
									batch_first=True)
			self.post_attn_fc1 = nn.Linear(self.K+self.action_embed_dim,  self.fc1_hidden_dim)
			self.post_attn_fc2 = nn.Linear( self.fc1_hidden_dim, self.flat_KN)

		self.ForwardOuput = namedtuple('ForwardOutput', [
													'pred_next_latent_belief', 
													'pred_next_latent_belief_logits'])

	def forward(self, latent_belief, action_embed, temp, gumbel_hard=False):
		self.gumbel_temperature = temp

		# ------------------------------------------------------------
		# Self-Attention of Factors
		# attention on N factors (how they interact with each other)
		# belief: [B, N, K]
		latent_belief =  latent_belief.squeeze().view(-1,self.N,self.K) # [B, N, K]
		attn_belief, _ = self.attention_belief(latent_belief, latent_belief, latent_belief) # [B, K, N]
		
		if self.use_belief_action_cross_attn:
			# aggregate attended belief across factors, [B, K]
			belief_aggregated = attn_belief.mean(dim=1)
			# ------------------------------------------------------------
			# Cross-Attention of self-attended belief and action
			# embed action
			# action_embed = self.ActionEmbedding(action)
			belief_for_cross_attn = belief_aggregated.unsqueeze(1) 	# [B, 1, K]
			action_for_cross_attn = action_embed.unsqueeze(1)		# [B, 1, action_embed_dim]
			attn_cross_belief_action, _ = self.attention_cross_belief_action(belief_for_cross_attn, action_for_cross_attn, action_for_cross_attn)

			# concat the results of self-attention and cross-attention
			attn_belief_action = torch.cat([belief_aggregated, attn_cross_belief_action.squeeze(1)], dim=-1)  # [B, K + action_embed_dim]
			belief_action = self.act_fn(self.post_attn_fc1(attn_belief_action))
			attn_belief = self.post_attn_fc2(belief_action)

		# ------------------------------------------------------------
		# Concat the self-attended belief with the action embedding and map back to some belief
		belief_action = torch.cat([attn_belief.reshape(-1, self.flat_KN), action_embed], dim=1) 
		h = self.act_fn(self.fc1(belief_action))
		h = self.act_fn(self.fc2(h))
		pred_next_latent_belief_logit = self.fc3(h) # [B,N*K]

		# reshape output from [B, F] -> [B, F, 1, 1]
		batch_size, latent_state_size = pred_next_latent_belief_logit.shape[0], pred_next_latent_belief_logit.shape[1]
		pred_next_latent_belief_logit = pred_next_latent_belief_logit.view(-1, self.N, self.K)
		pred_next_latent_belief = gumbel_softmax(
										   pred_next_latent_belief_logit.view(batch_size, latent_state_size, 1, 1), 
										   N=self.N, K=self.K, 
										   temp=self.gumbel_temperature, 
										   gumbel_hard=gumbel_hard)

		return self.ForwardOuput(
							pred_next_latent_belief, 
						  	pred_next_latent_belief_logit)