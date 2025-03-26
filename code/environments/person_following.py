from __future__ import annotations

import random 
import numpy as np

from agents.utils import a_star_planner
from environments.crossing import CrossingEnv
from environments.constants import Directions
from environments.person_of_interest import POI
from agents.pathfinding import PathfindingAgent

from minigrid.core.actions import Actions 
from minigrid.core.constants import (OBJECT_TO_IDX, DIR_TO_VEC)

class PersonFollowingEnv(CrossingEnv):
	'''
	Extending the Crossing environment to simulate a Person Following task
	'''
	def __init__(self, **kwargs): 
		super().__init__(**kwargs)

		self.poi = POI()
		self.poi_policy = PathfindingAgent(self.valid_actions,  mode='global')
		self.poi_policy.overlap_entities = {'empty','goal','ball','agent'} # overwrite to prevent plans that collide with agent
		self.poi_policy.colliders = {'wall'}  		# overwrite to prevent plans that collide with agent
		self.poi_policy.goal_entities = {'goal'}  	# overwrite to prevent plans that collide with agent
		self.poi_policy.ego_obj = 'ball'
		self.poi_pos = None 
		self.poi_dir = None 
		self.randommize_poi_dir = False

		self.default_poi_pos = (2,2)
		self.poi_to_far_away = 3 
		self.poi_distance_preference = 3
		self.consecutive_lost_steps = 0
		self.max_lost_steps = 3
		self.consecutive_waiting_steps = 0
		self.max_waiting_steps = 3
		
		self.goal_reached_multiplier = 100
		self.poi_reached_goal_reward = 100
		self.visibility_reward = 0.5
		self.staying_close_to_poi = 0.1
		self.poi_collision_penalty = -0.01
		self.poi_same_cell_penalty = -0.01

	def _gen_grid(self, width, height):
		super()._gen_grid(width, height)

	def place_poi(self):
		'''
		Place POI with the following contraints: 
		'''
		# always place POI next to the agent
		poi_pos = self.default_poi_pos
		self.poi.set_position(position=poi_pos)
		init_poi_dir = random.choice([d for d in Directions]) if self.randommize_poi_dir else Directions.right # default to start pointing right
		self.poi.set_direction(direction=init_poi_dir) 
		return self.put_obj(self.poi, *poi_pos)	 
	
	def poi_step(self, action):
		prev_poi_pos = self.poi.get_position()
		# Rotate left
		if action == Actions.left:
			new_dir = (self.poi.get_direction() - 1) % 4
			self.poi.set_direction(Directions(new_dir))

		# Rotate right
		elif action == Actions.right:
			new_dir = (self.poi.get_direction() + 1) % 4
			self.poi.set_direction(Directions(new_dir))

		# Move forward
		elif action == Actions.forward:

			# Get the position in front of the POI
			fwd_pos = self.poi.position + DIR_TO_VEC[self.poi.get_direction()]

			# Get the contents of the cell in front of the agent
			fwd_cell = self.grid.get(*fwd_pos)

			if (fwd_cell is None) or fwd_cell.can_overlap():
				# Set new position if it's valid
				self.poi.set_position(tuple(fwd_pos))
			
				poi_current_position = self.poi.get_position()
				self.grid.set(poi_current_position[0], poi_current_position[1], self.poi)
				self.grid.set(prev_poi_pos[0], prev_poi_pos[1], None)
		return 
	
	def get_poi_obs(self, include_poi):
		''' pathfinding agent requires observations to match the PriviledgedModelBuilder Wrapper'''
		return {'full_observation': self.get_grid(include_poi=include_poi), 'global_agent_direction':self.poi.get_direction()}
	
	def get_grid(self, include_agent=False, include_poi=False):
		env = self.unwrapped
		# only keep object_id dimension
		grid = env.grid.encode()[:,:,0]
		if include_agent:
			# include agent in view 
			grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([OBJECT_TO_IDX['agent']])
		if include_poi:
			poi_position = self.poi.get_position()
			grid[poi_position[0]][poi_position[1]] = np.array([OBJECT_TO_IDX[self.poi.type]])
		return grid
	
	def reset(self, seed=None, options=None):
		observations, info = super().reset(seed=seed, options=options)		
		self.consecutive_lost_steps = 0
		# place poi
		self.place_poi()
		# reset motion planner
		poi_obs = self.get_poi_obs(include_poi=True)
		self.poi_policy.reset(poi_obs)
		observations = self.gen_obs()
		return  observations, info 

	def step(self, action):
		obs, reward, terminated, truncated, info = super().step(action)

		# POI observations
		poi_obs = self.get_poi_obs(include_poi=True)
		assert np.argwhere(poi_obs['full_observation'] == OBJECT_TO_IDX[self.poi.type]).any(), \
			'POI (idx=6) is not in the grid'
		
		# POI step
		self._poi_step(poi_obs)

		# collision penalty
		reward += self.collision_penalty(action)

		# distance and visibility shaping 
		dist_to_poi = self.distance_to_poi()
		if dist_to_poi<=self.poi_to_far_away:
			self.consecutive_lost_steps = 0
			# standard distance_to_poi shaping
			if dist_to_poi > 0:  # there's a valid path to POI
				reward_factor = 1.5
				shaping_reward = reward_factor * (1.0 / (1.0 + dist_to_poi))
				reward += shaping_reward
			if self.agent_sees(*self.poi.get_position()):
				reward += self.visibility_reward
			# if staying close and POI reaches goal and agent sees POI at goal
			if self.poi_at_goal() and self.agent_sees(*self.poi.get_position()):
				reward += self.poi_reached_goal_reward
				terminated = True
		else:
			self.consecutive_lost_steps += 1
			reward -= 0.05 * dist_to_poi
			if self.consecutive_lost_steps >= self.max_lost_steps:
				truncated = True  # end the episode early for getting lost
			if self.poi_at_goal():
				truncated = True  # end the episode early for POI arriving to goal without escort

		# final obs
		obs = self.gen_obs()
		return obs, reward, terminated, truncated, info

	def agent_at_goal(self):
		if (self.agent_pos[0]==self.goal_position[0]) and (self.agent_pos[1]==self.goal_position[1]):
			return True
		else:
			return False

	def poi_at_goal(self):
		poi_position = self.poi.get_position()
		if (poi_position[0]==self.goal_position[0]) and (poi_position[1]==self.goal_position[1]):
			return True
		else:
			return False
	
	def collision_penalty(self, action):
		penalty = 0
		# Check if there is an obstacle in front of the agent
		front_cell = self.grid.get(*self.front_pos)

		collision = front_cell and front_cell.type == 'ball'
		# penalty if collision with POI (agent to close to POI)
		if (action == self.actions.forward) and (collision):
			penalty += self.poi_same_cell_penalty
		# if the are in the same cell
		poi_pos = self.poi.get_position()
		if (poi_pos[0] == self.agent_pos[0]) and (poi_pos[1] == self.agent_pos[1]):
			penalty += self.poi_collision_penalty
		return penalty
	
	def distance_to_poi(self):
		grid = self.grid.encode()[:,:,0]
		agent_dir = Directions(self.agent_dir)
		path, _ =  a_star_planner(grid=grid,
								 valid_actions=self.valid_actions,
								 init_direction=agent_dir,
								 start_pos=self.agent_pos,
								 goal_pos=self.poi.get_position(),
								 colliders=self.colliders_int,
								 goal_direction=self.poi.get_direction())
		
		# we could use the POI direction (goal_direction) in the planner
		# as adds one extra step in the plan
		
		if path is not None: 
			return len(path)
		else:
			return 0
		
	def distance_poi_to_goal(self):
		# returns the path length from the POI's position to the goal
		# 	using your existing A* pathfinder, or 0 if no path is found
		grid = self.grid.encode()[:,:,0]
		poi_pos = self.poi.get_position()
		path, _ = a_star_planner(
			grid=grid,
			valid_actions=self.valid_actions,
			init_direction=self.poi.get_direction(),
			start_pos=poi_pos,
			goal_pos=self.goal_position,
			colliders=self.colliders_int,
			goal_direction=None
		)
		if path is not None:
			return len(path)
		else:
			return 0

	def _poi_step(self, poi_obs):
		'''
		'look back' behavior for the POI:
		- If the agent doesn't see the POI or is too far, the POI might wait,
		up to self.max_waiting_steps in a row.
		- Otherwise, or once the wait limit is exceeded, the POI continues moving.
		'''

		waiting_condition = (not self.agent_sees(*self.poi.get_position()) or
							self.distance_to_poi() > self.poi_distance_preference)

		if waiting_condition:
			# POI waits
			self.consecutive_waiting_steps += 1
			if self.consecutive_waiting_steps < self.max_waiting_steps:
				return
		# if we are here, either waiting_condition is False (so no waiting),
		# or we've exceeded the wait limit and POI will move now
		self.consecutive_waiting_steps = 0

		poi_action = self.poi_policy.act(poi_obs)
		self.poi_step(poi_action)