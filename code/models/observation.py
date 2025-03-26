from collections import namedtuple

import torch
from torch import nn

from confs.definitions import ObservationModelConfig
from models.utils import gumbel_softmax, remap_obj_idx

class ObservationModel(nn.Module):
	def __init__(self, config: ObservationModelConfig, device='cpu'):
		super().__init__()
		self.model_name = 'ObservationModel'
		self.config = config
		self.init_gumbel_temperature = config.gumbel_temperature
		self.observation_dim = config.observation_dim
		self.concept_dim = config.concept_dim
		self.concept_embed_dim = config.concept_embed_dim
		self.action_embed_dim = config.action_embed_dim
		self.num_att_heads = config.num_att_heads
		self.K = config.categorical_dim # number of categories (factors in POMDP terms)
		self.N = config.num_categorical_distributions # dimension for each factor
		self.flat_KN = self.K*self.N
		self.conv1_hidden_dim =  config.conv1_hidden_dim
		self.prior_fc1_hidden_dim = 100
		self.object_idx_lookup = config.object_idx_lookup
		self.object_idx_rlookup = config.object_idx_rlookup
		self.prior_mode = config.prior_mode
		self.device = device

		# concept (object) in grid to vector embedding
		self.obs_to_concept_embedding = nn.Embedding(num_embeddings=self.concept_dim, embedding_dim=self.concept_embed_dim)
		
		# encoder
		self.enc_conv_1= nn.Conv2d(self.concept_embed_dim, self.conv1_hidden_dim, kernel_size=3, stride=1, padding=0)
		self.enc_conv_2= nn.Conv2d(self.conv1_hidden_dim, self.flat_KN, kernel_size=3, stride=1, padding=0)
		self.enc_att_1 = nn.MultiheadAttention(embed_dim=self.conv1_hidden_dim, num_heads=self.num_att_heads)

		# decoder
		# self.attention_belief = config.attention_belief_model
		self.attention_belief = nn.MultiheadAttention(embed_dim=self.K, num_heads=self.K, batch_first=True)
		self.dec_belief_action_fc1 = nn.Linear(self.flat_KN+self.action_embed_dim, 200)
		self.dec_belief_action_fc2 = nn.Linear(200, self.flat_KN)

		self.dec_conv1_1 = nn.ConvTranspose2d(self.flat_KN, self.conv1_hidden_dim, kernel_size=3, stride=1, output_padding=0, padding=0)
		self.dec_conv1_2 = nn.ConvTranspose2d(self.conv1_hidden_dim, self.concept_dim, kernel_size=3, stride=1, output_padding=0, padding=0)
		self.dec_att_1 =  nn.MultiheadAttention(embed_dim=self.concept_dim, num_heads=self.num_att_heads, batch_first=True)
		
		if self.prior_mode == 'prior_net':
			# learned prior
			self.prior_fc1 =  nn.Linear(self.N, self.prior_fc1_hidden_dim)
			self.prior_fc2 = nn.Linear(self.prior_fc1_hidden_dim, self.flat_KN)

		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
		self.act_fn = nn.ELU()

		self.temp = self.init_gumbel_temperature

		self.ForwardOuput = namedtuple('ForwardOutput', [
													'recon_next_obs_logits', 'recon_obs_logits', 
													'next_latent_belief', 'latent_belief',
													'prior_logits', 'next_prior_logits',
													'prior_probs', 'next_prior_probs',
													'latent_state_logits', 'next_latent_state_logits'])


	def encode(self, observation):
		obs_emb = self.obs_to_concept_embedding(observation).permute(0,3,1,2)
		h = self.act_fn(self.enc_conv_1(obs_emb))

		# attention
		batch_size, channels, height, width = h.size()
		h = h.view(batch_size, channels, height * width).permute(2, 0, 1)  # (sequence_length, batch_size, embedding_dim) for attention
		attn_output, _ = self.enc_att_1(h, h, h)
		attn_output = attn_output.permute(1, 2, 0).contiguous().view(batch_size, channels, height, width)

		latent_belief = self.act_fn(self.enc_conv_2(attn_output))

		return obs_emb, latent_belief
	
	def decode(self, latent_belief, action_embed=None):
		''' action dependent decoder with attention in the last layer '''
		# input latent_belief of shape [B, N*K, 1, 1]
		B = latent_belief.shape[0]
		
		# ------------------------------------------------------------
		# Self-Attention of Factors
		# attention on N factors (how they interact with each other)
		latent_belief =  latent_belief.squeeze().view(-1,self.N,self.K) # [B, N, K]
		attn_belief, _ = self.attention_belief(latent_belief, latent_belief, latent_belief) # [B, K, N]
		# ------------------------------------------------------------

		if action_embed is not None: # inject action to belief
			latent_belief_action = torch.cat([attn_belief.reshape(-1, self.flat_KN), action_embed], dim=1) # [B, K*N + action_embed_dim]
			h = self.act_fn(self.dec_belief_action_fc1(latent_belief_action))
			attn_belief = self.act_fn(self.dec_belief_action_fc2(h)) # [B, K, N]

		attn_belief = attn_belief.reshape(B, self.flat_KN, 1, 1) # [B, K*N, 1, 1]
		h = self.dec_conv1_1(attn_belief)
		recon_obs_logits = self.dec_conv1_2(h) # [B, concept_dim, width, height]

		# multi-head att expect shape (batch, seq, feature)
		recon_obs_logits = recon_obs_logits.view(-1,self.observation_dim*self.observation_dim,self.concept_dim)
		attn_recon_obs_logits, _ = self.dec_att_1(recon_obs_logits, recon_obs_logits, recon_obs_logits) # attn_belief: [B, 3x3, conv1_hidden_dim]
		attn_recon_obs_logits = attn_recon_obs_logits.reshape(-1, self.concept_dim, self.observation_dim, self.observation_dim)  # return: [B, concept_dim, obs_dim, obs_dim ]
		return attn_recon_obs_logits
	
	def _decode_mid_attn(self, latent_belief):
		''' NOT USED, kept here for documentation -- decode with attention in the middle layer '''
		# input latent_belief of shape [B, N*K, 1, 1]
		h = self.act_fn(self.dec_conv1_1(latent_belief))
		deconv_middle_layer_dim = 3 # hidden has: torch.Size([100, 4, 3, 3])
		h = h.view(-1, 3*3, self.conv1_hidden_dim)
		# multi-head att expect shape (batch, seq, feature)
		attn_belief, _ = self.dec_att_1(h, h, h) # attn_belief: [B, 3x3, conv1_hidden_dim]
		attn_belief = attn_belief.reshape(-1, self.conv1_hidden_dim, deconv_middle_layer_dim, deconv_middle_layer_dim) # attn_belief: [B, 3, 3, conv1_hidden_dim ]
		recon_obs_logits = self.dec_conv1_2(attn_belief)
		return recon_obs_logits
	
	def prior(self, context):
		''' input to the prior network could be used to bias the categories, or to provide context on what to sample'''
		h = self.act_fn(self.prior_fc1(context))
		prior_categorical_logits = self.act_fn(self.prior_fc2(h))
		return prior_categorical_logits
	
	def get_belief(self, observation, temp, gumbel_hard=False):
		obs_emb, latent_belief_logits = self.encode(observation)
		latent_belief = gumbel_softmax(latent_belief_logits, N=self.N, K=self.K, temp=temp, gumbel_hard=gumbel_hard)
		return latent_belief_logits, latent_belief
	
	def forward(self, observation, next_observation, action_embed, temp, gumbel_hard=False):
		self.temp = temp 

		latent_belief_logits, latent_belief = self.get_belief(observation, temp=temp, gumbel_hard=gumbel_hard)
		next_latent_belief_logits, next_latent_belief = self.get_belief(next_observation, temp=temp, gumbel_hard=gumbel_hard)

		recon_next_obs_logits = self.decode(next_latent_belief, action_embed)
		recon_obs_logits = self.decode(latent_belief, action_embed=None) # no action so keep latent intact

		if self.prior_mode == 'prior_net':
			# logits using prior 
			prior_logits = self.prior(torch.zeros(latent_belief.size(0), self.N, device=self.device))
			prior_logits = prior_logits.view(-1, self.N, self.K)
			prior_probs = gumbel_softmax(prior_logits, N=self.N, K=self.K, temp=self.temp, gumbel_hard=gumbel_hard)

			next_prior_logits = self.prior(torch.zeros(next_latent_belief.size(0), self.N, device=self.device))
			next_prior_logits = next_prior_logits.view(-1, self.N, self.K)
			next_prior_probs = gumbel_softmax(next_prior_logits, N=self.N, K=self.K, temp=self.temp, gumbel_hard=gumbel_hard)
		else:
			prior_logits, next_prior_logits = None, None
			prior_probs, next_prior_probs = None, None

		# reshape to match prior output shape
		latent_belief_logits = latent_belief_logits.view(-1, self.N, self.K)
		next_latent_belief_logits = next_latent_belief_logits.view(-1, self.N, self.K)

		return self.ForwardOuput(recon_next_obs_logits, recon_obs_logits, 
						   next_latent_belief, latent_belief, 
						   prior_logits, next_prior_logits, 
						   prior_probs, next_prior_probs,
						   latent_belief_logits, next_latent_belief_logits)