from collections import namedtuple, deque
import random 
import tqdm
from typing import Literal, Optional
import math 
import numpy as np 

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset, DataLoader

from minigrid.minigrid_env import MiniGridEnv

from models.world_model import WorldModel
from models.influence_estimator import InfluenceEstimator
from confs.definitions import BeliefDQNModelConfig, BeliefDQNModelTraining, WorldModelTrainingConfig
from helpers.metrics_utils import TrainingCallback, MetricTracker
from models.utils import (model_stats, 
						  observation_to_tensor,
						  env_data_to_tensors)

from models.world_model import update as world_model_update

Transition =namedtuple('Transition', ('observation', 'action', 'next_observation', 'rewards'))

class ReplayMemory(object):
	def __init__(self, capacity=1e6):
		self.memory = deque([], maxlen=capacity)
	def push(self, *args):
		''' push new transition '''
		self.memory.append(Transition(*args))
	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)
	def __len__(self):
		return len(self.memory)
	def __getitem__(self, index):
		return self.memory[index]

class DynamicReplayDataset(IterableDataset):
	def __init__(self, replay_buffer):
		'''
		Allows for treating the ReplayMemory as an iterable Dataset to be used by torch Dataloader
		'''
		self.replay_buffer = replay_buffer

	def __iter__(self):
		while True:
			# wait for at least one sample to be available.
			if len(self.replay_buffer) >= 0:
				# sample a batch from the memory buffer
				transitions = random.choice(self.replay_buffer)
				yield dict(transitions._asdict())
			else:
				raise ValueError('ReplayMemoery with {} entries does not have the requested number of samples: {}'.format(len(self.replay_buffer),self.sample_size))

class BeliefDQNModel(nn.Module):
	def __init__(self, config: BeliefDQNModelConfig, device='cpu', model_name='BeliefDQNModel', model_id='00'):
		super().__init__()
		self.model_name = model_name
		self.model_id  = model_id

		self.config = config

		self.num_influences = config.num_influences
		self.K = config.categorical_dim # number of categories (factors in POMDP terms)
		self.N = config.num_categorical_distributions # dimension for each factor
		self.flat_KN = self.K*self.N
		self.action_embed_dim = config.action_embed_dim
		self.num_actions = config.num_actions
		self.hidden_dim =  config.hidden_dim
		self.device = device

		self.fc_1 = nn.Linear((1+self.num_influences)*self.flat_KN, self.hidden_dim) # notice we have as inputs self + additional influences
		self.fc_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.fc_3 = nn.Linear(self.hidden_dim, self.num_actions)
		self.act_fn = nn.ReLU()

		self.ForwardOutput = namedtuple('ForwardOutput', ['state_action_value'])
	
	def forward(self, beliefs):
		# [batch, belief_id, N, K]
		B, belief_dim, N, K = beliefs.shape
		beliefs = beliefs.reshape(B, belief_dim*N*K) # reshape to [batch, belief_dim * N * K]
		h = self.act_fn(self.fc_1(beliefs))
		h = self.act_fn(self.fc_2(h))
		state_action_value = self.fc_3(h)

		return self.ForwardOutput(state_action_value)
	
	def act(self, beliefs):
		# max(1) returns the largest column value of each row
		#   second column on max is index of where max element was found
		#   we pick action with the largest expected reward
		state_action_value = self.forward(beliefs).state_action_value
		max_value_action = state_action_value.max(1).indices.view(-1)
		return max_value_action
	
	@classmethod
	def train(cls,
		   world_model: WorldModel,
		   model_config: BeliefDQNModelConfig,
		   training_config: BeliefDQNModelTraining,
		   env: MiniGridEnv,
		   training_metrics_callback: TrainingCallback,
		   device='cpu',
		   dynaq=False,
		   world_model_optimizer=None,
		   world_model_training_config: Optional[WorldModelTrainingConfig] = None):

		goal_completion_window = training_config.goal_completion_window

		eps_end = training_config.eps_end
		eps_start = training_config.eps_start
		eps_decay = training_config.eps_decay_coeff*training_config.num_episodes
		steps = 0

		sample_action = lambda: random.choice(list(model_config.valid_actions))

		policy_model = BeliefDQNModel(model_config, device=device, model_name='Policy', model_id=training_config.model_id).to(device)
		target_model = BeliefDQNModel(model_config, device=device, model_name='Target', model_id=training_config.model_id).to(device)
		memory = ReplayMemory(training_config.memory_size)
		optimizer = load_optimizer(policy_model, training_config)
		
		# initialize influence estimator 
		influence_estimator = InfluenceEstimator(model_config)

		# world model continued training
		if dynaq:
			world_model_memory = ReplayMemory(training_config.memory_size)
			world_model_dataset = DynamicReplayDataset(world_model_memory)
			world_model_data_loader = DataLoader(world_model_dataset, batch_size=world_model_training_config.batch_size)
			world_model_sampler = iter(world_model_data_loader)

		model_stats(policy_model)
		
		# initialize training metrics tracker, saving backup with prefix <ENV_NAME>_<INFLUENCE_MODE>
		metrics_tracker = MetricTracker(experiment_id='{}_{}'.format(env.unwrapped.__class__.__name__, training_config.influence_mode),backup_path='data/dqn_experiments/')
		writer = SummaryWriter(log_dir='logs/dqn/{}/{}/tensorboard/'.format(training_config.timestamp,training_config.experiment_name))
		
		train_pbar = tqdm.tqdm(range(training_config.num_episodes))

		goal_completion = []
		all_episode_rewards = []
		step_count = 0
		step_count_post_warmup = 0
		warmup_episodes = 0
		for episode in train_pbar:
			episode_rewards = []

			observation, _ = env.reset()
			with torch.no_grad(): 
				observation_tensor = observation_to_tensor(world_model.config.object_idx_lookup, observation, obs_key='observation', device=device)
				_, self_belief = world_model.get_belief(observation_tensor, gumbel_hard=False)
				influence = influence_estimator.estimate(env, world_model, observation, self_belief, influence_mode=training_config.influence_mode, device=device)

			beliefs = torch.stack([self_belief, influence], dim=1)
			
			done = False 
			reached_goal = False
			while not done:
				step_count += 1 
	
				if step_count_post_warmup==1:
					print('Size of memory after warm-up: {:,d} at episode {:,d}'.format(len(memory), episode))

				if training_config.warmup and step_count>training_config.warmup_steps: #TODO: need to fix this for when warmup is not used
					# post-warmup
					# decaying epsilon greedy
					post_warmpup_episode_num = episode-warmup_episodes
					eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * post_warmpup_episode_num / eps_decay)
					if random.random() > eps_threshold: # exploit
						with torch.no_grad():
							action = policy_model.act(beliefs)
					else: #explore
						action = torch.tensor([sample_action()], dtype=torch.long, device=device)
				else: 
					# warmp up random actions
					action = torch.tensor([sample_action()], dtype=torch.long, device=device)

				next_observation, rewards, terminated, truncated, _ = env.step(action.item())
				reward_tensor = torch.tensor([rewards], device=device)
				
				# prepare next observation to next belief 
				# with (torch.no_grad() if not dynaq else nullcontext()):  
				with torch.no_grad():  
					next_observation_tensor = observation_to_tensor(world_model.config.object_idx_lookup, next_observation, obs_key='observation', device=device)
					_, self_next_belief = world_model.get_belief(next_observation_tensor, gumbel_hard=False)
					next_influence = influence_estimator.estimate(env, world_model, next_observation, self_next_belief, influence_mode=training_config.influence_mode, device=device)
					next_beliefs = torch.stack([self_next_belief, next_influence], dim=1)

				if dynaq:
					world_model_memory.push(observation, action, next_observation, rewards)
					
					if (len(world_model_memory) > world_model_training_config.batch_size):
						
						batch = next(world_model_sampler)
						_observation = batch['observation']
						_action = batch['action']
						_next_observation = batch['next_observation']
						_rewards = batch['rewards']

						obj_remap_dict = world_model.config.object_idx_lookup

						(_observation_tensor, 
						_action_tensor, 
						_next_observation_tensor, 
						_rewards_tensor) = env_data_to_tensors(obj_remap_dict, 
																_observation, _action, 
																_next_observation, _rewards, 
																device=device)
						_action_tensor = _action_tensor.squeeze()
						batch_inputs = (_observation_tensor, _action_tensor, _next_observation_tensor, _rewards_tensor)

						_, _ = world_model_update(world_model, world_model_optimizer, world_model_training_config, batch_inputs, world_model.temp, device=device)
				
				# store the transition in memory
				memory.push(beliefs, action, next_beliefs, reward_tensor)
				
				done = truncated or terminated
				observation = next_observation
				beliefs = next_beliefs
				episode_rewards.append(rewards)

				if training_config.warmup and step_count>training_config.warmup_steps:
					step_count_post_warmup += 1
					if step_count_post_warmup % training_config.model_update_freq == 0: 
						loss = update(step_count_post_warmup, policy_model, target_model, optimizer, memory, training_config, device=device)
				else:
						loss = update(step_count, policy_model, target_model, optimizer, memory, training_config, device=device)

				steps += 1
				if terminated:
					next_beliefs = None
					reached_goal = True
				if done:
					break

			if training_config.warmup and step_count<=training_config.warmup_steps: 
				warmup_episodes += 1
			
			# goal completion tracking
			goal_completion.append(reached_goal)
			
			goal_completion_rate = np.mean(goal_completion[-goal_completion_window:]) if len(goal_completion) >= goal_completion_window else 0
		
			all_episode_rewards.append(sum(episode_rewards))
			average_episodic_rewards = sum(all_episode_rewards)/len(all_episode_rewards)

			metrics_tracker.track('cumulative_avg_episode_rewards', average_episodic_rewards, episode)
			metrics_tracker.track('goal_completion_rate', goal_completion_rate, episode)

			writer.add_scalar('Episode/Cumulative Avg. Episode Rewards', average_episodic_rewards, episode)
			writer.add_scalar('Episode/Goal Completion Rate', goal_completion_rate, episode)
			writer.add_scalar('Metrics/Memory Size', len(memory), episode)
			
			if (len(memory) > training_config.batch_size):
				if ((episode % 1) == 0):
							train_pbar.set_description('Training, Episode: [{}/{}] | Loss: {:.7f} -- AvgRewards: {:.3f} -- GoalCompletionRate: {:.3f}'\
										.format(episode+1, training_config.num_episodes, loss, average_episodic_rewards, goal_completion_rate))
				
				writer.add_scalar('Episode/Loss', loss.item(), episode)
				metrics_tracker.track('training_loss', loss.item(), episode)
			
		if not metrics_tracker.is_empty():
			training_metrics_callback('training_loss', metrics_tracker.get_epoch_average('training_loss'))
			training_metrics_callback('cumulative_avg_episode_rewards', metrics_tracker.get_episode_total_reward('cumulative_avg_episode_rewards'))
			training_metrics_callback('goal_completion_rate', metrics_tracker.get_epoch_average('goal_completion_rate'))
		
		print('Total Steps: {:,d}'.format(step_count))
		print('ReplayMemmory Size: {:,d}'.format(len(memory)))
		
		return policy_model, optimizer

def load_optimizer(model, config):
	return torch.optim.Adam(model.parameters(), lr=config.learning_rate)

def update(step:int,
		   policy_model, 
		   target_model, 
			optimizer,
			memory: ReplayMemory, 
			training_config: BeliefDQNModelTraining, 
			device='cpu',
			target_update_mode: Literal['soft', 'hard'] = 'soft'):
	''' runs a single learning step '''

	batch_size = training_config.batch_size

	if len(memory) < batch_size:
		return # not enough experiences
	
	optimizer.zero_grad()
	
	gamma = training_config.gamma
	tau = training_config.tau
	
	transition_sample = memory.sample(batch_size)
	# transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). 
	# 	this converts batch-array of Transitions to Transition of batch-arrays.
	sample_batch = Transition(*zip(*transition_sample))
	# compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after task ended)
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, sample_batch.next_observation)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in sample_batch.next_observation if s is not None])

	state_batch = torch.cat(sample_batch.observation)
	action_batch = torch.cat(sample_batch.action) # shape [B]
	reward_batch = torch.cat(sample_batch.rewards)

	# compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
	# 	these are the actions which would've been taken for each batch state according to the policy
	state_action_values = policy_model(state_batch).state_action_value
	state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1)).squeeze(1) # shape [B]

	# Double DQN: prevent the known to overestimate action values of standard DQN
	next_state_values = torch.zeros(batch_size, device=device)

	with torch.no_grad():
		# (1) select next_actions come from policy_model
		policy_output = policy_model(non_final_next_states)
		best_next_actions = policy_output.state_action_value.argmax(dim=1)

		# (2) evaluate using Q actions with target_model
		target_output = target_model(non_final_next_states)
		chosen_q_values = target_output.state_action_value.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

		# fill in non-final states
		next_state_values[non_final_mask] = chosen_q_values

	# Q-target:  r + gamma * Q_target
	expected_state_action_values = (next_state_values * gamma) + reward_batch
	loss = policy_loss(state_action_values, expected_state_action_values)

	loss.backward()
	param_list = list(policy_model.parameters())
	torch.nn.utils.clip_grad_norm_(param_list, training_config.grad_clip_norm, norm_type=2)
	optimizer.step()

	if target_update_mode == 'soft':
		# soft update of the target network's weights
		target_state_dict = target_model.state_dict()
		policy_state_dict = policy_model.state_dict()
		for key in policy_state_dict:
			target_state_dict[key] = policy_state_dict[key]*tau + target_state_dict[key]*(1-tau)
		target_model.load_state_dict(target_state_dict)
	elif target_update_mode == 'hard':
		# hard update: copy policy network parameters to target network at fixed intervals
		if step % training_config.target_update_freq == 0:
			target_model.load_state_dict(policy_model.state_dict())
	else:
		raise NotImplementedError('target_update_mode {} not implemented')
	
	return loss

def policy_loss(state_action_values, expected_state_action_values):
	huber_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='mean')
	return huber_loss