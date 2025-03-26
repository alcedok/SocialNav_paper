from __future__ import annotations

import itertools as itt
from typing import Tuple
import random 
import collections
import numpy as np

from agents.utils import a_star_planner
from environments.constants import Directions
from environments.augmented_grid import AugmentedGrid

# from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, COLORS, DIR_TO_VEC

class CrossingEnv(MiniGridEnv):
	"""
	## Description

	Modified version of CrossingEnv from MiniGrid, main changes:
	- rendering functions 
	- enabling/disabling of see_through_walls


	Depending on the `obstacle_type` parameter:
	- `Lava` - The agent has to reach the green goal square on the other corner
		of the room while avoiding rivers of deadly lava which terminate the
		episode in failure. Each lava stream runs across the room either
		horizontally or vertically, and has a single crossing point which can be
		safely used; Luckily, a path to the goal is guaranteed to exist. This
		environment is useful for studying safety and safe exploration.
	- otherwise - Similar to the `LavaCrossing` environment, the agent has to
		reach the green goal square on the other corner of the room, however
		lava is replaced by walls. This MDP is therefore much easier and maybe
		useful for quickly testing your algorithms.

	## Mission Space
	Depending on the `obstacle_type` parameter:
	- `Lava` - "avoid the lava and get to the green goal square"
	- otherwise - "find the opening and get to the green goal square"

	## Action Space

	| Num | Name         | Action       |
	|-----|--------------|--------------|
	| 0   | left         | Turn left    |
	| 1   | right        | Turn right   |
	| 2   | forward      | Move forward |
	| 3   | pickup       | Stay in place|
	| 4   | drop         | Unused       |
	| 5   | toggle       | Unused       |
	| 6   | done         | Unused       |

	## Observation Encoding

	- Each tile is encoded as a 3 dimensional tuple:
		`(OBJECT_IDX, COLOR_IDX, STATE)`
	- `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
		[minigrid/core/constants.py](minigrid/core/constants.py)
	- `STATE` refers to the door state with 0=open, 1=closed and 2=locked

	## Rewards

	A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

	## Termination

	The episode ends if any one of the following conditions is met:

	1. The agent reaches the goal.
	2. The agent falls into lava.
	3. Timeout (see `max_steps`).

	## Registered Configurations

	S: size of the map SxS.
	N: number of valid crossings across lava or walls from the starting position
	to the goal

	- `Lava` :
		- `MiniGrid-LavaCrossingS9N1-v0`
		- `MiniGrid-LavaCrossingS9N2-v0`
		- `MiniGrid-LavaCrossingS9N3-v0`
		- `MiniGrid-LavaCrossingS11N5-v0`

	- otherwise :
		- `MiniGrid-SimpleCrossingS9N1-v0`
		- `MiniGrid-SimpleCrossingS9N2-v0`
		- `MiniGrid-SimpleCrossingS9N3-v0`
		- `MiniGrid-SimpleCrossingS11N5-v0`

	"""

	def __init__(self, **config):  
		self.__dict__.update(config)
		self.num_valid_actions = len(self.valid_actions)
		self.ego_agent_pos = (self.agent_view_size // 2, self.agent_view_size - 1)

		self.colliders = {'wall'}

		mission_space = MissionSpace(mission_func=self._gen_mission)

		
		super().__init__(
			mission_space=mission_space,
			width=self.width,
			height=self.height,
			see_through_walls=self.see_through_walls,
			max_steps=self.max_steps,
			agent_view_size=self.agent_view_size,
			render_mode = self.render_mode)

		self.action_space.n = self.num_valid_actions
		
		assert self.num_valid_actions==self.action_space.n, \
			'Number of valid_actions {} != Size of action_space {}'.format(self.num_valid_actions, self.action_space.n)

	@property
	def colliders_int(self):
		return [OBJECT_TO_IDX[o] for o in self.colliders]

	@staticmethod
	def _gen_mission():
		return "find the opening and get to the green goal square."
	
	def get_reachable_positions(self, start_pos=(1,1)):
		''' 
		Performs search to find all reachable positions from start_pos. 
		Crossing logic seems to always have (1,1) be the origin for all paths, so we keep as default
		'''
		reachable = set()
		queue = collections.deque([start_pos])
		reachable.add(start_pos)

		while queue:
			current_pos = queue.popleft()
			x, y = current_pos

			for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # explore cardinal neighbors
				next_x, next_y = x + dx, y + dy
				next_pos = (next_x, next_y)

				if (0 < next_x < self.width - 1 and 0 < next_y < self.height - 1 and # within world bounds
					self.grid.get(next_x, next_y) is None and # empty cell
					next_pos not in reachable): # not already visited
					reachable.add(next_pos)
					queue.append(next_pos)
		return reachable

	def get_valid_positions(self, grid, 
						 start_x, start_y, 
						 goal_x, goal_y,
						 distance_ratio, tolerance, colliders_int, 
						 discard_start=False, discard_goal=False):
		
		candidate_positions = self.get_reachable_positions()
		candidate_positions_set = set(candidate_positions)
		if discard_goal:
			candidate_positions_set.discard((goal_x, goal_y))	# remove goal
		if discard_start:
			candidate_positions_set.discard((start_x, start_y)) # remove start
		candidate_positions = list(candidate_positions_set)
		
		# if start and goal are the same, choose an option from candidates
		if (start_x, start_y) == (goal_x, goal_y):
			# then choose a random candidate as the target goal
			(goal_x, goal_y) = random.choice(candidate_positions)

		# get maximum distance using A*
		optimal_max_path, _ =  a_star_planner(grid=grid,
								 valid_actions=self.valid_actions,
								 init_direction=Directions.right,
								 start_pos=(start_x, start_y),
								 goal_pos=(goal_x, goal_y),
								 colliders=colliders_int,
								 goal_direction=None)
		
		assert optimal_max_path, 'something went wrong when calculating valid agent positions, optimal_max_path={}'.format(optimal_max_path)
		max_distance = len(optimal_max_path)

		desired_distance_from_goal = int((distance_ratio) * max_distance)


		valid_positions = []
		for pos in candidate_positions:
			optimal_path, _ =  a_star_planner(grid=grid,
								 valid_actions=self.valid_actions,
								 init_direction=Directions.right,
								 start_pos=pos,
								 goal_pos=(goal_x, goal_y),
								 colliders=colliders_int,
								 goal_direction=None)
			
			distance = len(optimal_path)
			if abs(distance - desired_distance_from_goal) <= tolerance:
				valid_positions.append(pos)
		return (valid_positions, candidate_positions, desired_distance_from_goal)
	
	def place_agent(self, goal_x, goal_y):
		# default start position 
		start_x, start_y = (1,1)

		self.agent_pos = (1,1)
		self.agent_dir = 0

	def _gen_grid(self, width, height):
		assert width % 2 == 1 and height % 2 == 1  # odd size

		# Create an empty grid
		self.grid =AugmentedGrid(width, height)
		

		# Generate the surrounding walls
		self.grid.wall_rect(0, 0, width, height)

		# Place a goal square in the bottom-right corner
		goal_x, goal_y = width - 2, height - 2
		self.put_obj(Goal(), goal_x, goal_y)
		self.goal_position = (goal_x, goal_y)

		# Place obstacles (lava or walls)
		v, h = object(), object()  # singleton `vertical` and `horizontal` objects
		rivers = [(v, i) for i in range(2, height - 2, 2)] + [(h, j) for j in range(2, width - 2, 2)]
		self.np_random.shuffle(rivers)
		rivers = rivers[:self.num_crossings]  # sample random rivers
		rivers_v = sorted(pos for direction, pos in rivers if direction is v)
		rivers_h = sorted(pos for direction, pos in rivers if direction is h)
		obstacle_pos = itt.chain(
			itt.product(range(1, width - 1), rivers_h),
			itt.product(rivers_v, range(1, height - 1)),
		)
		for i, j in obstacle_pos:
			self.put_obj(self.obstacle_type(), i, j)

		# Sample path to goal
		path = [h] * len(rivers_v) + [v] * len(rivers_h)
		self.np_random.shuffle(path)

		# Create openings
		limits_v = [0] + rivers_v + [height - 1]
		limits_h = [0] + rivers_h + [width - 1]
		room_i, room_j = 0, 0
		for direction in path:
			if direction is h:
				i = limits_v[room_i + 1]
				j = self.np_random.choice(range(limits_h[room_j] + 1, limits_h[room_j + 1]))
				room_i += 1
			elif direction is v:
				i = self.np_random.choice(range(limits_v[room_i] + 1, limits_v[room_i + 1]))
				j = limits_h[room_j + 1]
				room_j += 1
			else:
				assert False
			self.grid.set(i, j, None)

		self.mission = "find the opening and get to the green goal square"

		# place agent
		self.place_agent(goal_x, goal_y)

	'''
	Rendering
	'''

	def show_render(self):
		import matplotlib.pylab as plt	
		plt.imshow(self.render())
		plt.axis('off')
	
	def obs_to_image(self, obs):
		grid, _ = self.grid.decode(obs)
		image = grid.render(tile_size=self.tile_size, 
						agent_pos=self.ego_agent_pos, 
						agent_dir=3)
		return image
	
	def partial_to_full_obs(self, obs, poi_local_directions=None):
		'''
		must be in batches of observations [B,N,N]
		this is very hacky, but used just for visualization purposes without having to refactor Minigrid directly
		
		poi_local_directions: flat list of Directions from the agent's coordinate frame
		'''

		## color dimension
		color_dim = np.ones_like(obs)
		env_obj_colors = {'wall':'grey', 'unseen':'grey', 'door':'blue' ,'goal':'green', 'ball':'blue'}

		for obj_name, obj_color in env_obj_colors.items():
			obj_id = OBJECT_TO_IDX[obj_name]
			color_id = COLOR_TO_IDX[obj_color]
			color_dim[obs==obj_id] = color_id
		
		# state dimension
		state_dim = np.zeros_like(obs)
		# where is the door
		door_positions = np.argwhere(obs == OBJECT_TO_IDX['door'])
		if len(door_positions)!=0:
			for (obs_idx, row, col) in door_positions:
				cells_to_keep_door_open = {
					(self.ego_agent_pos[0], self.ego_agent_pos[1]-1),
					(self.ego_agent_pos[0], self.ego_agent_pos[1])
					} #Minigrid uses (col, row)
				if (row,col) in cells_to_keep_door_open:
					# door_pos
					state_dim[obs_idx,row,col] =STATE_TO_IDX['open']
				else: 
					state_dim[obs_idx,row,col] =STATE_TO_IDX['closed']
		
		# managing of POI direction thru the ball object on the state dimension
		# assumes we are running with PriviledgedModelBuilder Wrapper and the AugmentedGrid
		ball_positions = np.argwhere(obs == OBJECT_TO_IDX['ball'])
		if len(ball_positions)!=0:
			for (obs_idx, row, col) in ball_positions:
				state_dim[obs_idx,row,col] = poi_local_directions[obs_idx]

		# combine dims
		full_dim = np.stack([obs,color_dim,state_dim],axis=-1)
		return full_dim
	
	def reset(self, seed=None, options=None):
		observations, info = super().reset(seed=seed, options=options)
		return observations, info 
	
	def step(self, action):
		observations, reward, terminated, truncated, info = super().step(action)
		return observations, reward, terminated, truncated, info 