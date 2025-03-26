import tqdm
import datetime, time, os 
import numpy as np
from collections import namedtuple
from dataclasses import asdict
from typing import Literal
import random 

import torch
from torch import nn
from torch.utils.data import DataLoader

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv

from environments.constants import Directions
from agents.abstract import Agent
from models.observation import ObservationModel
from models.action import ActionModel
from models.transition import TransitionModel
from models.inverse_model import InverseModel
from confs.definitions import WorldModelTrainingConfig, WorldModelConfig
from helpers.metrics_utils import TrainingCallback, MetricTracker
from helpers.data_utils import collect_experiences
from models.utils import (world_model_loss, 
						  compute_class_weights, 
						  env_data_to_tensors, 
						  training_reconstruction_accuracy, 
						  uniform_latent_sampler,
						  learned_prior_sampler,
						  observation_logits_to_labels,
						  model_stats)

class WorldModel(nn.Module):
	def __init__(self, 
			  config: WorldModelConfig,
			  model_id='00',
			  device='cpu'):
		super().__init__()

		#configs
		self.config = config
		self.observation_model_config = config.observation_model_config
		self.action_model_config = config.action_model_config
		self.transition_model_config = config.transition_model_config
		self.inverse_model_config = config.inverse_model_config

		self.model_id  = model_id
		
		# hyper-parameters 
		self.temp = config.gumbel_temperature
		self.gumbel_hard = config.gumbel_hard
		self.categorical_dim = config.categorical_dim
		self.num_categorical_distributions = config.num_categorical_distributions
		self.action_embed_dim = config.action_embed_dim
		self.belief_dim = self.categorical_dim*self.num_categorical_distributions
		self.proposed_class_weights = None 
		self.prior_mode = config.prior_mode
		self.K = self.categorical_dim # number of categories (factors in POMDP terms)
		self.N = self.num_categorical_distributions # dimension for each factor
		self.flat_KN = self.K*self.N

		# models
		self.observation_model = ObservationModel(self.observation_model_config, device=device).to(device)
		self.action_model = ActionModel(self.action_model_config, device=device).to(device)
		self.transition_model = TransitionModel(self.transition_model_config, device=device).to(device)
		self.inverse_model = InverseModel(self.inverse_model_config, device=device).to(device)

		# used when saving and loading the world model checkpoint 
		self.model_list = [self.observation_model, self.action_model, self.transition_model, self.inverse_model]

		self.ForwardOutout = namedtuple('ForwardOutput', ['observation_model_output', 'action_model_output', 'transition_model_output', 'pred_recon_next_obs_from_latent_belief', 'inverse_model_output'])
		self.LossOutput = namedtuple('LossOutput', ['total_loss', 'observation_loss', 'transition_loss', 'observation_recon_loss', 'observation_kld', 'inverse_loss'])

	def freeze_weights(self):
		all_internal_models  = nn.ModuleList(self.model_list)
		for model in all_internal_models:
			# set to eval
			model.eval()
			for param in model.parameters():
				# freeze weights
				param.requires_grad = False 
	
	def get_belief(self, observation, gumbel_hard=False):
		# batch dimension is required for downstream, we check for it and add if needed
		# note this assumes that shape must be [B, N, N], where N is agent_view_size from environment
		if len(observation.shape) != 3:
			observation = observation.unsqueeze(0)
		latent_belief_logits, latent_belief = self.observation_model.get_belief(observation, temp=self.temp, gumbel_hard=gumbel_hard)
		return latent_belief_logits.view(-1, self.N, self.K), latent_belief.view(-1, self.N, self.K)
	
	def forward(self, observation, next_observation, action, temp, gumbel_hard=False):
		# action embed
		action_model_output = self.action_model(action)
		action_embed = action_model_output.action_embed
		
		# observation components
		observation_model_output = self.observation_model(observation, next_observation, action_embed, temp=temp, gumbel_hard=gumbel_hard)
		belief = observation_model_output.latent_belief

		# transition components
		transition_model_output = self.transition_model(belief, action_embed, temp, gumbel_hard=gumbel_hard)
		next_belief = transition_model_output.pred_next_latent_belief
		
		# inverse model components
		inverse_model_output = self.inverse_model(belief, next_belief)

		# reconstruct the predicted next_belief 
		pred_recon_next_obs_from_latent_belief = self.observation_model.decode(next_belief, action_embed)

		world_model_output = self.ForwardOutout(observation_model_output, action_model_output, transition_model_output, pred_recon_next_obs_from_latent_belief, inverse_model_output)
		return world_model_output
	
	@classmethod
	def train(cls, 
			 model_config: WorldModelConfig, 
			 training_config: WorldModelTrainingConfig,
			 env: MiniGridEnv, 
			 agent: Agent,
			 training_metrics_callback: TrainingCallback,
			 device='cpu'):
		
		obj_remap_dict = model_config.object_idx_lookup
		model = WorldModel(model_config, device=device, model_id=training_config.model_id)
		experience_buffer = collect_experiences(env, agent,training_config)

		if training_config.compute_proposed_class_weights:
			model.proposed_class_weights = compute_class_weights(experience_buffer=experience_buffer, mapping_dict=obj_remap_dict).to(device)
			print('Proposed Class Weights: ', model.proposed_class_weights, 'size:', model.proposed_class_weights.shape)

		train_dataloader = DataLoader(experience_buffer, batch_size=training_config.batch_size, shuffle=True)
		dataset_size = len(train_dataloader)

		optimizer = load_optimizer(model, training_config)
		learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=training_config.learning_rate_gamma)
		
		# print model stats and param info
		model_stats(model)

		# initialize training metrics tracker
		metrics_tracker = MetricTracker()

		cur_temperature = training_config.initial_temperature

		for epoch in range(training_config.epochs):

			train_pbar = tqdm.tqdm(train_dataloader)

			for batch_idx, sample_batch in enumerate(train_pbar):
				(episode, step, observation, action, next_observation, rewards, terminated, truncated) = sample_batch
				# convert environment sample data to tensors
				(observation_tensor, action_tensor, next_observation_tensor, rewards_tensor) = env_data_to_tensors(obj_remap_dict, observation, action, next_observation, rewards, device=device)
				batch_inputs = (observation_tensor, action_tensor, next_observation_tensor, rewards_tensor)

				# update model
				loss_output, world_model_output = update(model, optimizer, training_config, batch_inputs, cur_temperature, device=device)
	

				if ((batch_idx % 10) == 0):
					train_pbar.set_description('Training, Epoch: [{}/{}] | Total loss: {:.7f} -- O_recon: {:.7f} --  O_kld: {:.7f} -- T_loss: {:.7f} -- I_loss: {:.7f}'\
								.format(epoch+1,
										training_config.epochs,
										loss_output.total_loss,
										loss_output.observation_recon_loss,
										loss_output.observation_kld,
										loss_output.transition_loss,
										loss_output.inverse_loss))


				# calculate epoch-level eval recon accuracy metrics 
				correct_next_obs_recon_batch, correct_obs_recon_batch, correct_pred_obs_recon_batch	= training_reconstruction_accuracy(next_observation_tensor, observation_tensor, world_model_output, device=device)

				# epoch-level metrics
				metrics_tracker.track('training_loss', loss_output.total_loss.item(), epoch, batch_idx)
				metrics_tracker.track('observation_loss', loss_output.observation_loss.item(), epoch, batch_idx)
				metrics_tracker.track('transition_loss', loss_output.transition_loss.item(), epoch, batch_idx)
				metrics_tracker.track('inverse_loss', loss_output.inverse_loss.item(), epoch, batch_idx)
				metrics_tracker.track('next_obs_recon_accuracy', correct_next_obs_recon_batch, epoch, batch_idx)
				metrics_tracker.track('obs_recon_accuracy', correct_obs_recon_batch, epoch, batch_idx)
				metrics_tracker.track('pred_obs_recon_accuracy', correct_pred_obs_recon_batch, epoch, batch_idx)

			# print epoch level reconstruction accuracy report
			total_cell_count = (len(experience_buffer)*(observation_tensor.shape[1]*observation_tensor.shape[2]))
			print('NextObs acc: {:.2f}% -- Obs acc: {:.2f}% -- predObs acc: {:.2f}% \n'.format(\
				metrics_tracker.get_epoch_recon_accuracy('next_obs_recon_accuracy', epoch, total_cell_count), 
				metrics_tracker.get_epoch_recon_accuracy('obs_recon_accuracy', epoch, total_cell_count), 
				metrics_tracker.get_epoch_recon_accuracy('pred_obs_recon_accuracy', epoch, total_cell_count)))

			# incrementally anneal temperature and learning rate if enabled
			if ((epoch % 1) == 0) and (training_config.temp_anneal):
				cur_temperature = np.maximum(training_config.initial_temperature*np.exp(-training_config.temperature_anneal_rate*epoch), training_config.minimum_temperature)
				
			learning_rate_scheduler.step() # multiply learning rate by learning_rate_gamma
			print('\tupdated temperature: {:.3f}'.format(cur_temperature))
			print('\tcurrent learning rate: {:.3e}'.format(learning_rate_scheduler.get_last_lr()[0])) 

		training_metrics_callback('training_loss', metrics_tracker.get_epoch_average('training_loss'))
		training_metrics_callback('observation_loss', metrics_tracker.get_epoch_average('observation_loss'))
		training_metrics_callback('transition_loss', metrics_tracker.get_epoch_average('transition_loss'))
		training_metrics_callback('inverse_loss', metrics_tracker.get_epoch_average('inverse_loss'))

		# final model setup 
		model.temp = cur_temperature

		return model, optimizer

	
def update(model, optimizer, training_config, inputs, temp, device='cpu'):
	#TODO: need to account for dynamic loss weights like kld 
	''' runs a single learning step '''
	optimizer.zero_grad()

	param_list = list(model.parameters())

	(observation, action, next_observation, rewards) = inputs

	world_model_output = model(observation, next_observation, action, temp, gumbel_hard=model.gumbel_hard)

	# compute loss
	(o_recon_loss, t_loss, o_kld, action_embed_mse) = world_model_loss(
											model.config,
											next_observation, observation, world_model_output,
											prior_mode=model.prior_mode,
											proposed_class_weights=model.proposed_class_weights, 
											device=device)
	
	o_kld = o_kld * training_config.kl_loss_weight
	o_loss = o_recon_loss + o_kld
	i_loss = action_embed_mse
	loss = o_loss + t_loss + i_loss

	loss.backward()
	torch.nn.utils.clip_grad_norm_(param_list, training_config.grad_clip_norm, norm_type=2)
	optimizer.step()

	loss_output = model.LossOutput(loss, o_loss, t_loss, o_recon_loss, o_kld, i_loss)

	return loss_output, world_model_output

def load_optimizer(model, config):
	return torch.optim.Adam(model.parameters(), lr=config.initial_learning_rate)

def load_model_from_checkpoint(training_config, model_id, checkpoint_file, epoch, root_save_path='data/models', device='cpu', frozen=False):
	# get checkpoint
	checkpoint = torch.load(os.path.join(root_save_path, model_id, checkpoint_file), map_location=device)

	# instantiate model based on config
	config = WorldModelConfig(**checkpoint['world_model_config'])
	model = WorldModel(config, model_id=model_id, device=device).to(device)
	# load all the weights
	model.load_state_dict(checkpoint['world_model_state_dict'])

	for sub_model in model.model_list:
		name = sub_model.model_name
		sub_model.load_state_dict(checkpoint['sub_models'][name]['state_dict'])
	
	# load optimizer
	optimizer = load_optimizer(model, training_config)
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	if frozen:
		model.freeze_weights()

	model_devices_check = set(param.device for param in model.parameters())
	print('Model checkpoint loaded on device {}'.format(model_devices_check))
	return model, optimizer, epoch

def save_model_checkpoint(model, optimizer, epoch, root_save_path='data/models'):
	'''
	Save model weights
	'''

	# collection of models that make up the world model
	sub_models_dict = {
		sub_model.model_name: {
			'state_dict':sub_model.state_dict(),
			'config':asdict(sub_model.config)} for sub_model in model.model_list
	}
	
	# complete meta-model 
	world_model_dict = {
		'world_model_state_dict': model.state_dict(),
		'world_model_config': asdict(model.config),
		'world_model_proposed_class_weights': model.proposed_class_weights,
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
		'sub_models': sub_models_dict
	}

	path_to_model = os.path.join(root_save_path, model.model_id)
	if not os.path.isdir(path_to_model):
			os.makedirs(path_to_model)

	datetime_ = datetime.datetime.now().strftime("%Y%m%d")
	seconds_ = int(time.time()) 
	model_filename = 'checkpoint_{}-{}.pth'.format(datetime_,seconds_)
	model_filepath = os.path.join(path_to_model, model_filename)
	print('Saving Model: \n\t{}'.format(model_filepath))
	torch.save(world_model_dict, model_filepath)
	return model_filename,  model.model_id

def sample_from_latent(model, action_list, 
					   prior_type: Literal['uniform', 'prior'], 
					   remap_dict: dict,
					   device='cpu', convert_output_to_array=True, hard=False):

	SampleOutputs = namedtuple('SampleOutputs', ['latent', 'next_latent', 
											  	 'observation','next_observation',
												 'action_list','poi_local_directions'])
	cur_temp = model.temp
	print('using temperature', cur_temp)

	# sample a batch random states (B, N, K): batch, num_distributions, num_categories
	num_samples = len(action_list)
	B = num_samples
	N =  model.config.num_categorical_distributions
	K = model.config.categorical_dim
	observation_dim = model.observation_model.observation_dim
	latent_shape = (B, N ,K)
	
	poi_local_directions = []

	random_actions_tensor = torch.tensor(action_list, device=device, dtype=torch.int)

	if prior_type == 'uniform':
		# sample using uniform
		states = uniform_latent_sampler(latent_shape, 
										temperature=cur_temp, 
										hard=hard).view(-1, N*K).view(-1, N*K, 1, 1).to(device)
	elif prior_type == 'learned':
		# use learn, ensure no gradients are accumulated
		with torch.no_grad():
			prior_logits = model.observation_model.prior(torch.zeros(B, N, device=device)).view(-1, N, K)
			states = learned_prior_sampler(prior_logits, 
								  temperature=cur_temp, 
								  hard=hard).view(-1, N*K).view(-1, N*K, 1, 1).to(device)
	else:
		raise NotImplementedError('prior_type {} is not currently supported.'.format(prior_type))
	
	with torch.no_grad():

		observation_logits = model.observation_model.decode(states)
		observation = observation_logits_to_labels(observation_logits, remap_dict, device=device)
		observation = observation.reshape(num_samples, observation_dim, observation_dim)
		latent = states.reshape(num_samples, K, N)
		random_actions_embed = model.action_model(random_actions_tensor).action_embed

		transition_model_output = model.transition_model(states, random_actions_embed, cur_temp, gumbel_hard=hard)
		next_states = transition_model_output.pred_next_latent_belief.squeeze().view(-1, N*K, 1, 1)
		next_latent = next_states.reshape(num_samples, K, N)

		next_obs_logits = model.observation_model.decode(next_states, action_embed=random_actions_embed)
		next_observation = observation_logits_to_labels(next_obs_logits, remap_dict, device=device)
		next_observation = next_observation.reshape(num_samples, observation_dim, observation_dim)

		# since we don't currently encode POI direction we randomize it for rendering here
		# TODO: we should explicitly model POI direction in the future
		poi_local_directions = random.choices(population=[d for d in Directions], k=B) # sample B number of directions
		print('poi_local_directions', poi_local_directions)
	
	if convert_output_to_array: 
		latent = latent.cpu().numpy()
		next_latent = next_latent.cpu().numpy()
		observation = observation.cpu().numpy()
		next_observation = next_observation.cpu().numpy()

	return SampleOutputs(latent, next_latent, observation, next_observation, action_list, poi_local_directions)