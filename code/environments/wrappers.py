import numpy as np
import copy
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX
from environments.constants import Directions

class PartiallyObservable(gym.ObservationWrapper):
	'''
	Egocentric, partially observable gridworld.
	Unlike the raw observations from MiniGrid we only use the entity layer (ignore color and state)

	Sensing range defined by 'agent_view_size', 
	where the agent only perceives in-front and to the sides, 
	see diagram below for agent_view_size=3:
	
	+---+---+---+
	|   |   |   |
	+---+---+---+
	|   |   |   |
	+---+---+---+
	|   | ▲ |   |
	+---+---+---+
	 
	'''

	def __init__(self, env, agent_view_size):
		super().__init__(env)
		assert agent_view_size % 2 == 1
		assert agent_view_size >= 3

		self.agent_view_size = agent_view_size
	
		view_space = spaces.Box(
			low=0,
			high=255,
			shape=(self.env.unwrapped.width, self.env.unwrapped.height, 1),
			dtype='uint8')

		self.observation_space = spaces.Dict({
			'observation': view_space})

	def observation(self, obs):
		env = self.unwrapped
		grid, vis_mask = env.gen_obs_grid(self.agent_view_size)
		
		# encode the partially observable view into a numpy array
		# only keep object_id dimension
		view = grid.encode(vis_mask)[:,:,0]

		return {'observation': view}


class FullyObservable(gym.ObservationWrapper):
	'''
	Allocentric, fully observable gridworld using a compact grid encoding.
	
	Grid is represented as a matrix where each (row,col) corresponds
	to a position (x,y) in the grid. The value at (i,j) is the object index. 

	The global 'direction' of the agent is provided as a separate entity in the observation dict. 	
	'''

	def __init__(self, env):
		super().__init__(env)
		
		state_space = spaces.Box(
			low=0,
			high=255,
			shape=(self.env.unwrapped.width, self.env.unwrapped.height, 1),
			dtype='uint8')

		self.observation_space = spaces.Dict({
			'state': state_space,
			'agent_direction': spaces.Discrete(4)
			})

	def observation(self, obs):
		env = self.unwrapped
		# only keep object_id dimension
		state = env.grid.encode()[:,:,0]
		
		# include agent in view 
		state[env.agent_pos[0]][env.agent_pos[1]] = np.array([OBJECT_TO_IDX['agent']])
		# state[env.agent_pos[1]][env.agent_pos[0]] = np.array([OBJECT_TO_IDX['agent']]) # incorrect?

		return {
			'state': state, 
			'agent_direction':Directions(obs['direction'])}

class PriviledgedModelBuilder(gym.ObservationWrapper):
	'''
	This is a priviledged wrapper since it provides both full and partial observability.
	We use this when using the expert planner for initial data collection. 

	Egocentric: partially observable gridworld.
	Allocentric: fully observable gridworld using a compact grid encoding.
	
	Grid is represented as a matrix where each (row,col) corresponds
	to a position (x,y) in the grid. The value at (i,j) is the object index. 

	The global 'direction' of the agent is provided as a separate entity in the observation dict. 	
	'''

	def __init__(self, env, agent_view_size):
		super().__init__(env)
		
		self.fully_observable_wrapper = FullyObservable(env)
		self.partially_observable_wrapper = PartiallyObservable(env, agent_view_size)
		
		self.observation_space = spaces.Dict({
			'full_observation': self.fully_observable_wrapper.observation_space['state'],
			'partial_observation': self.partially_observable_wrapper.observation_space['observation'],
			'agent_direction': self.fully_observable_wrapper.observation_space['agent_direction']
			})

	def get_relative_direction(self, agent_global_direction: Directions, poi_global_direction: Directions) -> Directions:
		'''
		compute the relative direction of POI from the agent's reference frame
		'''
		rotation_steps = 0
		if agent_global_direction == Directions.right:
			rotation_steps = 1
		elif agent_global_direction == Directions.down:
			rotation_steps = 2
		elif agent_global_direction == Directions.left:
			rotation_steps = 3

		relative_direction_index = (poi_global_direction.value - rotation_steps) % len(Directions)
		return Directions(relative_direction_index)

	def observation(self, obs):
		env = self.unwrapped

		full_obs = self.fully_observable_wrapper.observation(obs)
		partial_obs = self.partially_observable_wrapper.observation(obs)
		partial_obs_with_poi = copy.deepcopy(self.partially_observable_wrapper.observation(obs))
		
		if getattr(env, 'poi', False):
			global_poi_direction = env.unwrapped.poi.get_direction()
			local_poi_direction = self.get_relative_direction(full_obs['agent_direction'], global_poi_direction)
			partial_obs['observation'][partial_obs['observation'] == OBJECT_TO_IDX['ball']] = OBJECT_TO_IDX['empty'] # remove POI from local view
		else: 
			global_poi_direction = -1 # torch Dataloaders can't deal with None values, so we use -1 as a placeholder
			local_poi_direction = -1

		return {
			'observation': partial_obs['observation'], # keep for compatability with other sub-systems downstream
			'full_observation': full_obs['state'],
			'partial_observation': partial_obs['observation'],
			'partial_observation_with_poi': partial_obs_with_poi['observation'],
			'global_agent_direction':full_obs['agent_direction'],
			'local_agent_direction':Directions.up,
			'global_poi_direction':global_poi_direction,
			'poi_local_direction':local_poi_direction}