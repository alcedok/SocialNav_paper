from typing import Literal

from models.utils import (observation_to_tensor, action_to_tensor, uniform_latent_sampler)
from agents.pathfinding import PathfindingAgent
from environments.utils import (gen_obs_at_point)

from minigrid.minigrid_env import MiniGridEnv

import torch

class InfluenceEstimator:
	'''
	Implementation of the Perspective-Shift Operator
	'''
	def __init__(self, config):
		self.config = config
		self.valid_actions = config.valid_actions
		self.observation_dim = config.observation_dim # same as agent_view_size
		self.see_through_walls = config.see_through_walls
		
		self.K = config.categorical_dim
		self.N = config.num_categorical_distributions

		self.last_subgoal = None
		self.last_known_belief = None
		self.poi_mode = True if config.goal_entities == 'ball' else False
		self.planner_mode = 'local' if self.poi_mode else 'global'
		self.operator = PathfindingAgent(self.valid_actions,  mode=self.planner_mode, agent_view_size=self.observation_dim)
		self.goal_entities = self.set_goal_entities(config.goal_entities) # currently only support one goal type

	def set_goal_entities(self, goal_entities):
		# check if goal entities need to be changed
		goal_entities = {goal_entities} # convert string to set, since we only support one right now
		if self.operator.goal_entities != goal_entities:
			self.operator.goal_entities = goal_entities
		if 'ball' in self.operator.goal_entities:
			self.operator.observation_grid_key = 'partial_observation_with_poi'
		return goal_entities

	def get_subgoal(self,env: MiniGridEnv, observation):	
		''' calculate the local view subgoal in allocentric coordinates'''
		plan_subgoals = self.operator.get_subgoals() # sequence of all subgoals to goal

		if not plan_subgoals: # if there are no subgoals use the last one, this happens when we reached the goal
			return self.last_subgoal

		if self.planner_mode == 'global':
			# if using global planner we already have allocentric coordinates
			# keep the subgoal that is furthest away from the local view	
			allocentric_subgoal = None
			step_count = 0 # steps to subgoal, including current position
			for (x,y,dir) in plan_subgoals:
				# if subgoal not in view then keep the one before
				in_view = env.unwrapped.in_view(x,y)
				if not in_view:
					break
				allocentric_subgoal = (x,y,dir)
				step_count += 1

		elif self.planner_mode == 'local':
			# when planner_mode is local this only gets called in perfect-information mode
			poi_global_pos = env.unwrapped.poi.get_position()
			poi_global_dir = env.unwrapped.poi.get_direction()
			allocentric_subgoal = (poi_global_pos[0], poi_global_pos[1], poi_global_dir) # (x, y, dir)
			step_count = 0 # not used in this mode but kept here for consistency with the return
		else: 
			raise NotImplementedError('planner_mode [{}] not supported in InfluenceEstimator'.format(self.planner_mode)) 

		return allocentric_subgoal, step_count
	
	def estimate(self, 
			  	env: MiniGridEnv,
				world_model, 
				observation: dict,
				self_belief: torch.Tensor,
				influence_mode: Literal['zero', 'random', 'perfect-information', 'perspective-shift'],
				device='cpu'):
		
		# batch dimension is required for downstream, we check for it and add if needed
		# note this assumes that shape must be [B, N, N], where N is agent_view_size from environment
		observation_tensor = observation_to_tensor(world_model.config.object_idx_lookup, observation, obs_key='observation', device=device)
		if len(observation_tensor.shape) != 3:
			observation_tensor = observation_tensor.unsqueeze(0)
		batch_size = observation_tensor.shape[0]
		latent_shape = (batch_size, self.N, self.K)

		if self.last_known_belief is None: # if we haven't seen the goal yet, we initialize
			self.last_known_belief = uniform_latent_sampler(latent_shape, temperature=world_model.temp, hard=world_model.gumbel_hard, device=device)

		# -----------------------------------------
		# handle influence modes
		if influence_mode == 'zero':
			influence_belief = torch.zeros(size=latent_shape, device=device)

		elif influence_mode == 'random':
			influence_belief = uniform_latent_sampler(latent_shape, temperature=world_model.temp, hard=world_model.gumbel_hard, device=device)

		elif influence_mode == 'perfect-information':
			self.operator.reset(observation) # generate a new plan to the goal
			
			# if there is no plan then we return the last known belief 
			if not self.operator.goal_found:
				influence_belief = self.last_known_belief 
			
			# otherwise we extract the belief at the subgoal 
			else: 
				env_grid = env.get_wrapper_attr('grid') # grid object from MiniGridEnv
				allocentric_subgoal, action_step_count = self.get_subgoal(env, observation) # visible subgoal (x,y,direction) in allocentric coordinates
				allocentric_subgoal_pos = (allocentric_subgoal[0], allocentric_subgoal[1])
				allocentric_subgoal_dir = allocentric_subgoal[2]

				self.last_subgoal = allocentric_subgoal

				# returns a new partially observable Grid object
				obs_at_goal, _ = gen_obs_at_point(cur_pos=allocentric_subgoal_pos, 
														cur_dir=allocentric_subgoal_dir, 
														grid=env_grid, 
														agent_view_size=self.observation_dim, 
														see_through_walls=self.see_through_walls)
				
				# we only keep the OBJ_IDX layer as per most of our approach in this work
				obs_at_goal = obs_at_goal.encode()[:,:,0]
				obs_at_goal = {'observation':obs_at_goal} # to match the environment data structure for observations
				
				# convert new observation to belief
				obs_at_goal_tensor = observation_to_tensor(world_model.config.object_idx_lookup, obs_at_goal, obs_key='observation', device=device)
				_, influence_belief = world_model.get_belief(obs_at_goal_tensor, gumbel_hard=False)
			
		elif influence_mode == 'perspective-shift':
			self.operator.reset(observation, goal_direction=observation['poi_local_direction']) # generate a new plan to the goal

			# if there is no plan then we return the last known belief
			if not self.operator.goal_found:
				influence_belief = torch.zeros(size=latent_shape, device=device)
			
			# otherwise we compute the belief at the subgoal iteratively
			else: 
				action_plan = self.operator.get_plan()
				
				# if we are running on global mode we need to chop the global plan to only those within visual range
				if self.planner_mode == 'global':
					allocentric_subgoal, action_step_count = self.get_subgoal(env, observation) # visible subgoal (x,y,direction) in allocentric coordinates
					action_plan = action_plan[:action_step_count-1]
					self.last_subgoal = allocentric_subgoal

				# rollout forward the current belief to the goal position using the action plan
				cur_belief = self_belief
				for action in action_plan:
					action_tensor = action_to_tensor(action, device).unsqueeze(0) # [B, actions]
					action_embed = world_model.action_model(action_tensor).action_embed
					transition_output = world_model.transition_model(cur_belief, action_embed, temp=world_model.temp, gumbel_hard=world_model.gumbel_hard)
					cur_belief = transition_output.pred_next_latent_belief
				influence_belief = cur_belief.view(batch_size, self.N, self.K)
		else:
			raise NotImplementedError('influence_mode {} not implemented'.format(influence_mode))
		
		self.last_known_belief = influence_belief
		return influence_belief
	
