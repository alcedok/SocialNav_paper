import os
import math
import numpy as np
from collections import Counter, namedtuple
import matplotlib.pylab as plt

from environments.constants import Directions

import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.distributions as dist
import torch.nn.functional as F
import torch.utils.data

def model_stats(model):
	total_model_params = sum(p.numel() for p in model.parameters())
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print('Total Trainable Params {:,d}'.format(total_trainable_params))
	print('Total Params {:,d}'.format(total_model_params))

def compute_class_weights(experience_buffer, mapping_dict=None):
	objects = []
	# collect all objects from observations
	for sample in experience_buffer:
		obs_img_obj = sample.observation['observation'].flatten().tolist()
		next_obs_img_obj = sample.next_observation['observation'].flatten().tolist()
		objects.extend(obs_img_obj)
		objects.extend(next_obs_img_obj)

	# count object frequencies 
	obj_count = Counter(objects)
	# if using a object_index map then map objects to corresponding new index
	if mapping_dict is not None:
		obj_count = {mapping_dict[k]:v for k,v in obj_count.items()}
		num_classes = len(mapping_dict.keys()) # use num classes given the expected map
	else:
		num_classes = len(obj_count.keys()) # otherwise use the num of objects that actuall appear in dataset
	
	# initialize class weights
	proposed_weights = [1.0]*num_classes

	# compute inverse frequency ratio'ed by the max count
	max_class_count = obj_count[max(obj_count, key=obj_count.get)]
	for obj in obj_count.keys():
		w = max_class_count/obj_count[obj]
		proposed_weights[obj] = proposed_weights[obj]*w

	# proposed trick by taking square root 
	proposed_class_weights = torch.tensor(np.sqrt(proposed_weights).tolist())
	return proposed_class_weights

		
def world_model_loss(model_config, next_observation, observation,
					 world_model_output,
					 proposed_class_weights=None, 
					 prior_mode='uniform',
					 device='cpu'):
	
	B = observation.shape[0]
	N = model_config.num_categorical_distributions
	K = model_config.categorical_dim

	##--------------------------- 
	# observation loss components
	cross_entropy_next_obs = compute_ce_loss(world_model_output.observation_model_output.recon_next_obs_logits,
										   next_observation, weights=proposed_class_weights)
	cross_entropy_obs = compute_ce_loss(world_model_output.observation_model_output.recon_obs_logits, 
									 observation, weights=proposed_class_weights)

	latent_belief = world_model_output.observation_model_output.latent_belief.reshape(B, N, K)
	next_latent_belief = world_model_output.observation_model_output.next_latent_belief.reshape(B, N, K)

	##--------------------------- 
	## transition loss components

	# transition reconstruction loss
	t_loss = compute_ce_loss(world_model_output.pred_recon_next_obs_from_latent_belief, next_observation, weights=proposed_class_weights)
	
	## MSE between logits of next_latent (target) and pred_next_latent (predicted)
	# pred_next_latent_logits = world_model_output.transition_model_output.pred_next_latent_belief_logits.reshape(B, -1)
	# next_latent_logits = world_model_output.observation_model_output.next_latent_state_logits.reshape(B, -1)
	# t_loss += F.mse_loss(pred_next_latent_logits, next_latent_logits.detach(), reduction='mean')

	## KLD between categorical distirbution of next_latent (target) and pred_next_latent (predicted)
	pred_next_latent_belief = world_model_output.transition_model_output.pred_next_latent_belief.reshape(B, N, K)
	t_loss += belief_state_kl_divergence(next_latent_belief.detach(), pred_next_latent_belief) # transition output is the model to follow
	
	##--------------------------- 
	## inverse model loss components
	true_action_embedding = world_model_output.action_model_output.action_embed
	pred_action_embedding = world_model_output.inverse_model_output.pred_action_emb
	action_embed_mse = F.mse_loss(pred_action_embedding, true_action_embedding.detach(), reduction='mean') # could use detach() on the true embedding

	##--------------------------- 
	## prior KLD
	if prior_mode == 'prior_net':
		# kld between the predicted latent and the learned prior, prior being the target distribution
		prior_probs = world_model_output.observation_model_output.prior_probs.reshape(B, N, K)
		kld_loss = compute_learned_prior_divergence(latent_belief, prior_probs)
		# the prior being the target distribution
		next_prior_probs = world_model_output.observation_model_output.next_prior_probs.reshape(B, N, K)
		kld_loss += compute_learned_prior_divergence(pred_next_latent_belief, next_prior_probs)
	elif prior_mode == 'uniform':
		# kld between the predicted latent and uniform prior
		kld_loss = compute_uniform_prior_divergence(latent_belief, device=device)
		# kld_loss +=  compute_uniform_prior_divergence(pred_next_latent_belief,  device=device)
	else: 
		raise NotImplementedError('KLD computation using prior_mode {} not implemented'.format(prior_mode))

	##---------------------------
	# Total 
	ce_total = (cross_entropy_next_obs + cross_entropy_obs)
	kl_o_total = kld_loss

	return ce_total, t_loss, kl_o_total, action_embed_mse

def compute_learned_prior_divergence(p_probs, q_probs):
	# p: predicted (data)
	# q: target (model)
	B, N, K = q_probs.shape

	q_probs = q_probs.view(B*N, K)
	p_probs = p_probs.view(B*N, K)
	
	# epsilon to avoid log(0) issues
	eps = 1e-8
	p_probs = p_probs.clamp(min=eps)
	q_probs = q_probs.clamp(min=eps)

	q = dist.Categorical(probs=(q_probs))
	p = dist.Categorical(probs=(p_probs))

	# kl is of shape [B*N]
	kl = dist.kl.kl_divergence(p, q) 
	kl = kl.view(B, N)
	
	kl_loss = torch.mean(
				torch.mean(kl, dim=1))
	return kl_loss

def compute_uniform_prior_divergence(p_probs, device='cpu'):
	# p: predicted (data)
	# q: target (model: uniform)

	# probs of shape [B, N, K] where B is batch, N is number of categorical distributions, K is number of classes
	# in our case it is of the shape [B,1,8]
	B, N, K = p_probs.shape
	
	p_probs = p_probs.view(B*N, K)

	# epsilon to avoid log(0) issues
	eps = 1e-8
	p_probs = p_probs.clamp(min=eps)
	
	p = dist.Categorical(probs=(p_probs))
	# uniform bunch of K-class categorical distributions
	q = dist.Categorical(probs=torch.full((B*N, K), 1.0/K).to(device))
	# kl is of shape [B*N]
	kl = dist.kl.kl_divergence(p, q)
	kl = kl.view(B, N)
	
	kl_loss = torch.mean(
				torch.mean(kl, dim=1))
	return kl_loss

def belief_state_kl_divergence(pred_belief, belief):
	# inputs are probs of shape [B, N, K] where B is batch, N is number of categorical distributions, K is number of classes
	# p: predicted
	# q: target
	B, N, K = belief.shape
	pred_belief = pred_belief.view(B*N, K)
	belief = belief.view(B*N, K)
	# epsilon to avoid log(0) issues
	eps = 1e-8
	pred_belief = pred_belief.clamp(min=eps)
	belief = belief.clamp(min=eps)

	# predicted distribution
	p = dist.Categorical(probs=(pred_belief))
	# target distribution
	q = dist.Categorical(probs=(belief))
	# kl is of shape [B*N]
	kl = dist.kl.kl_divergence(p, q) 
	kl = kl.view(B, N)
	
	kl_loss = torch.mean(
				torch.mean(kl, dim=1))
	return kl_loss

def compute_ce_loss(recon_obs_logits, obs, weights=None, use_focal_loss=False, alpha=1, gamma=1):
	obs_target = obs.flatten(start_dim=1).long()
	obs_pred = recon_obs_logits.flatten(start_dim=2)
	if use_focal_loss:
		'''
		alpha: weighting factor for class imbalance.	
		gamma:  focusing parameter that adjusts the rate at which easy examples are down-weighted.
		'''
		cross_entropy_obs = F.cross_entropy(obs_pred, obs_target, reduction='none', weight=weights)
		pt = torch.exp(-cross_entropy_obs)
		focal_loss = alpha * (1 - pt) ** gamma * cross_entropy_obs
		return focal_loss.mean()
	else:
		cross_entropy_obs = F.cross_entropy(obs_pred, obs_target, reduction='mean', weight=weights)
	
	return cross_entropy_obs

def count_correct_pred(actual_obs, recon_obs_logits, device='cpu'):
	obs_recon_pred_labels = observation_logits_to_labels(recon_obs_logits, device=device)
	# element-wise equality check
	obs_eq = torch.eq(actual_obs.flatten(start_dim=1).long().squeeze(), obs_recon_pred_labels)
	# sum correct 
	correct_obs_recon = torch.sum(obs_eq)
	return correct_obs_recon, obs_recon_pred_labels, obs_eq

def training_reconstruction_accuracy(next_observation, observation, world_model_output, device='cpu'):
	# next obs recon
	correct_next_obs_recon_batch, _, _ = count_correct_pred(next_observation, world_model_output.observation_model_output.recon_next_obs_logits, device)
	# obs recon
	correct_obs_recon_batch, _, _ =  count_correct_pred(observation, world_model_output.observation_model_output.recon_obs_logits, device)
	# pred_next_obs recon (using transition from obs and action)
	correct_pred_obs_recon_batch, _, _ =  count_correct_pred(next_observation, world_model_output.pred_recon_next_obs_from_latent_belief, device)

	return correct_next_obs_recon_batch, correct_obs_recon_batch, correct_pred_obs_recon_batch

def gumbel_softmax(latent_belief, N, K, temp, gumbel_hard=False):
		# latent_state is of shape: [B batch, N*K feature, 1, 1]
		# reshape to be able to sample N distributions 
		latent_belief_NK = latent_belief.view(-1, N, K)
		# take the gumbel-softmax on the last dimension K (dim=-1)
		latent_belief_categorical_NK = F.gumbel_softmax(latent_belief_NK, tau=temp, hard=gumbel_hard, dim=-1)
		# reshape back to original shape
		latent_belief_categorical = latent_belief_categorical_NK.view(latent_belief.shape)
		return latent_belief_categorical

def compression_factor(image_shape: tuple, latent_shape: tuple):
	'''
	compression factor of representation
	'''
	numerator = math.prod(image_shape)
	denominator = math.prod(latent_shape)
	return numerator/denominator

## ------------------------------------------------------------------
## Tensor formating, reshaping and loading

def to_tensor(data, dtype, device):
		if not torch.is_tensor(data):
			return torch.tensor(data, dtype=dtype, device=device)
		return data.to(dtype=dtype, device=device)

def observation_to_tensor(obj_remap_dict, observation, obs_key='observation', device='cpu'):
	# keep this separate because we re-use in other parts of the code
	observation_tensor = to_tensor(observation[obs_key], torch.int, device)
	# remap object indices to the range of concept dimensions
	observation_tensor_remapped = remap_obj_idx(observation_tensor, obj_remap_dict, device)
	return observation_tensor_remapped

def action_to_tensor(action, device):
	# keep this separate because we re-use in other parts of the code
	return to_tensor(action, torch.int, device)

def env_data_to_tensors(obj_remap_dict, observation, action, next_observation, rewards, device='cpu'):
	''' convert data to tensors if not already, and remap observation object index '''
	observation_tensor = observation_to_tensor(obj_remap_dict, observation, device=device)
	action_tensor = action_to_tensor(action, device)
	next_observation_tensor = observation_to_tensor(obj_remap_dict, next_observation, device=device)
	rewards_tensor = to_tensor(rewards, torch.float, device)
	return (observation_tensor, action_tensor, next_observation_tensor, rewards_tensor)

def remap_obj_idx(input_tensor, mapping_dict, device):
	''' remap input integer tensor with a corresponding lookup '''
	max_num_keys = (max(mapping_dict.keys()) + 1)
	mapping_tensor = torch.full( (max_num_keys,), -1, dtype=torch.int, device=device) # create mapping tensor with set size from lookup
	for old_val, new_val in mapping_dict.items():
		mapping_tensor[old_val] = new_val
	output_tensor = mapping_tensor[input_tensor]
	return output_tensor

## ------------------------------------------------------------------

def observation_logits_to_labels(recon_obs_logits, remap_dict=None, device='cpu'):
	# logits -> probs, and reshape
	obs_recon_probs = F.softmax(recon_obs_logits.flatten(start_dim=2), dim=1)
	# probs -> label (highest prob), and reshape
	obs_recon_to_labels = torch.max(obs_recon_probs, dim=1, keepdim=True).indices.squeeze()#.squeeze(1)
	# remap back to original object indices from minigrid
	if remap_dict is not None:
		obs_recon_to_labels = remap_obj_idx(obs_recon_to_labels, remap_dict, device=device)
	return obs_recon_to_labels

def uniform_latent_sampler(latent_shape: tuple, temperature: float, hard:bool = False, device='cpu'):
	''' sample from a uniform prior categorical distribution using gumbel-softmax '''
	if not isinstance(latent_shape, tuple) or len(latent_shape) != 3:
		raise ValueError('Expected shape to be a tuple of length 3 (B, N, K), received {} '.format(latent_shape))
	uniform_prior = torch.zeros(size=latent_shape, device=device)+1e-20
	sample = F.gumbel_softmax(uniform_prior, tau=temperature, hard=hard, dim=-1)
	return sample

def learned_prior_sampler(prior_logits: torch.Tensor, temperature: float, hard: bool = False):
	''' sample from a learned prior categorical distribution using gumbel-softmax
		prior logits of the learned prior distribution, shape (B, N, K); used for context
	'''
	if not isinstance(prior_logits, torch.Tensor) or len(prior_logits.shape) != 3:
		raise ValueError('Expected prior_logits to be a tensor of shape (B, N, K), received {}'.format(prior_logits.shape))
	
	# apply Gumbel-Softmax sampling
	sample = F.gumbel_softmax(prior_logits, tau=temperature, hard=hard, dim=-1)
	return sample

def mental_vs_real(env, model, policy, device='cpu', sim_steps=None, use_poi_dir_goal=False, root_save_path=None):
	'''
	Simulate navigating to a target configuration from an observation.
	Re-using a lot of methods from other parts, making sure its consistent

	input:
	- environment 
	- World Model: require Observation, Transition
	- policy: an agent that will navigate to specified point

	return:
	- the real trajectory observations
	- the simulated trajectory decoded observations
	- the actual step-by-step latent from observations
	- the simulated step-by-step latent  
	'''
	MentalSimulationOutput = namedtuple('MentalSimulationOutput', ['observations', 
																	'pred_latent_states_reconstruction', 
																	'latent_states', 
																	'pred_latent_states',
																	'actions',
																	'poi_local_directions'])
	
	DataSample = namedtuple('DataSample', 	['step',
											'observation', 'action', 
											'next_observation', 'rewards',
											'terminated', 'truncated'])
	
		
	obj_idx_map = model.config.object_idx_lookup
	obj_idx_remap = model.config.object_idx_rlookup

	model_temp = model.temp
	model_gumbel_hard = model.gumbel_hard

	# collect actual observations
	observation, _ = env.reset()
	env.show_render()

	if root_save_path is not None:
		save_path = os.path.join(root_save_path,model.model_id)
		os.makedirs(save_path, exist_ok=True)
		save_path = os.path.join(save_path,'global.jpg')
		plt.savefig(save_path, bbox_inches='tight')
		print('\nSaved global env render in:\n\t', save_path)
		plt.close()
		
	if use_poi_dir_goal:
		poi_dir = observation['poi_local_direction'] 
	else: 
		poi_dir = None 

	policy.reset(observation, goal_direction=poi_dir) 		# generate action plan
	actions = policy.get_plan().copy()
	if sim_steps is None:
		sim_steps = len(actions)		# set the max number of simulation to the num of actions
	# print('actions:', actions)
	trajectory = []
	for step in range(sim_steps):
		action = policy.act(observation)
		next_observation, rewards, terminated, truncated, _ = env.step(action)
		sample = DataSample(step, observation, action, next_observation, rewards, terminated, truncated)
		trajectory.append(sample)
		observation = next_observation

	# collect trajectory into a single batch	
	trajectory_dataloader = DataLoader(trajectory, batch_size=sim_steps, shuffle=False)

	poi_local_directions = []
	with torch.no_grad():
		for batch in trajectory_dataloader:
			(step, observation, action, next_observation, rewards, terminated, truncated) = batch

			(observation_tensor, 
			action_tensor, 
			next_observation_tensor, _) = env_data_to_tensors(
												obj_idx_map, 
												observation, action, next_observation, rewards, 
												device=device)
			
			world_model_output = model(observation_tensor, next_observation_tensor, action_tensor, temp=model_temp, gumbel_hard=model_gumbel_hard)
			
			# encode all observations into latents
			latent_states = world_model_output.observation_model_output.latent_belief

			# apply transition to the first latent state
			cur_belief = latent_states[0:1] # the first latent, use step+1 to keep the batch dim
			pred_latent_states_list = []
			for step in range(sim_steps):
				cur_action = action_tensor[step:step+1] 
				cur_action_embed = model.action_model(cur_action).action_embed
				transition_output = model.transition_model(cur_belief, cur_action_embed, temp=model_temp, gumbel_hard=model_gumbel_hard)
				cur_belief = transition_output.pred_next_latent_belief
				pred_latent_states_list.append(cur_belief)
			
			# decode pred latent to observation
			action_embed = model.action_model(action_tensor).action_embed
			pred_latent_states = torch.cat(pred_latent_states_list, dim=0)
			pred_latent_states_recon_logits = model.observation_model.decode(pred_latent_states, action_embed)
			pred_latent_states_recon = observation_logits_to_labels(pred_latent_states_recon_logits, obj_idx_remap, device=device)

			# collect poi directions for visualization
			poi_dir_flat_list_cur_obs = observation['poi_local_direction'].cpu().numpy().tolist()
			poi_dir_flat_list_nxt_obs = next_observation['poi_local_direction'].cpu().numpy().tolist()[-1:]
			poi_local_directions.extend(poi_dir_flat_list_cur_obs)
			poi_local_directions.extend(poi_dir_flat_list_nxt_obs)
			
	# reshape for plotting 
	B = sim_steps
	N =  model.config.num_categorical_distributions
	K = model.config.categorical_dim
	V = model.observation_model.observation_dim
	latent_shape = (B+1, K ,N) # include the last observation latent
	recon_shape = (B+1, V, V)
	
	# include last observation that was collected in next_observation_tensor
	observation_tensor =  torch.cat([observation['observation'], next_observation['observation'][-1].unsqueeze(0)], dim=0)
	# observation_tensor =  torch.cat([observation['partial_observation_with_poi'], next_observation['partial_observation_with_poi'][-1].unsqueeze(0)], dim=0)
	# include the latent from the last observation that was collected in next_observation_tensor
	last_obs_latent = world_model_output.observation_model_output.next_latent_belief[-1].unsqueeze(0)
	latent_states = torch.cat([latent_states, last_obs_latent], dim=0)

	# include an empty tensor in the reconstrucion for the first observation 
	empty_tensor = torch.zeros(size=(1,V*V), device=device)
	pred_latent_states_recon = torch.cat([empty_tensor, pred_latent_states_recon], dim=0)

	# include an empty tensor in the pred latent for the first observation
	empty_tensor = torch.zeros(size=(1,K*N,1,1), device=device)
	pred_latent_states = torch.cat([empty_tensor, pred_latent_states], dim=0)

	observations = observation_tensor.cpu().numpy()
	latent_states = latent_states.reshape(latent_shape).cpu().numpy()
	pred_latent_states = pred_latent_states.reshape(latent_shape).cpu().numpy()
	pred_latent_states_recon = pred_latent_states_recon.reshape(recon_shape,).cpu().numpy()

	return MentalSimulationOutput(observations, pred_latent_states_recon, latent_states, pred_latent_states, actions, poi_local_directions)