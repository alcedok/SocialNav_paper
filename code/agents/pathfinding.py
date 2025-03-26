import random
import numpy as np
from typing import Literal

from agents.abstract import Agent
from agents.utils import a_star_planner, heuristic, SearchConfiguration
from environments.constants import Directions, DirectionsXY

from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.actions import Actions

# TODO: only works with entity matrix, not the complete observation/state tensor from minigrid

class PathfindingAgent(Agent):
	'''
	A pathfinding agent that explores environment using local or global observations depending on mode.
	In global mode, we use A* pathfinding to the goal

	In local mode, this agent is unlikely to find the goal, but provides experiences that may be useful.
	attempt to generate a plan for each cell starting from the ones furthest away;
		skips the egocentric cell, leave it to the end as the fallback
	'''
	def __init__(self, 
			  valid_actions, 
			  mode: Literal['global','local'],
			  agent_view_size=None,
			  agent_name='pathfinding-agent'):
		
		if (mode == 'local') and (agent_view_size is None):
			raise ValueError('when using local mode, agent_view_size must be included; agent_view_size is {}'.format(agent_view_size))
		
		self.valid_actions = valid_actions
		self.agent_name = agent_name
		self.mode = mode 
		self.agent_view_size = agent_view_size

		# A* planner config params
		self.current_plan = None
		self.current_goal = None
		self.current_sub_goals = None
		self.goal_found = None

		self.ego_obj = 'agent'
		self.colliders = {'wall'}
		self.overlap_entities = {'empty','goal','agent'}
		self.goal_entities = {'goal'} # currently only support one goal type

		if self.mode == 'local':
			self.observation_grid_key = 'partial_observation'
			self.observation_dir_key = 'local_agent_direction'

		elif self.mode == 'global':
			self.observation_grid_key = 'full_observation'
			self.observation_dir_key = 'global_agent_direction'
		else:
			raise NotImplementedError('mode {} not implemented.'.format(self.mode))

	@property
	def ego_obj_int(self):
		return OBJECT_TO_IDX[self.ego_obj]

	@property
	def colliders_int(self):
		return [OBJECT_TO_IDX[o] for o in self.colliders]

	@property
	def overlap_entities_int(self):
		return [OBJECT_TO_IDX[o] for o in self.overlap_entities]
	
	@property
	def goal_entities_int(self):
		return [OBJECT_TO_IDX[o] for o in self.goal_entities]

	def get_plan(self):
		return self.current_plan
	
	def get_goal(self):
		return self.current_goal
	
	def get_subgoals(self):
		return self.current_sub_goals
	
	def get_agent_pos(self, grid):
		if self.mode == 'local':
			agent_pos = (self.agent_view_size // 2, self.agent_view_size - 1) # constants when running on local mode

		elif self.mode == 'global':
			agent_pos = np.argwhere(grid == OBJECT_TO_IDX[self.ego_obj])[0]
			agent_pos = tuple(agent_pos.tolist())
		else:
			raise NotImplementedError('mode {} not implemented.'.format(self.mode))
		return agent_pos
	
	def set_goal_found(self, value):
		self.goal_found = value 

	def reset(self, observation, goal_direction=None):
		grid = observation[self.observation_grid_key]
		agent_pos = self.get_agent_pos(grid)
		agent_dir = observation[self.observation_dir_key]
		self.current_plan, self.current_goal, self.current_sub_goals = self.generate_plan(grid, agent_pos, agent_dir, goal_direction=goal_direction)
		
	def act(self, observation, goal_direction=None):
		# continue acting until plan is finished
		# if plan is finished then replan based on obs
		if len(self.current_plan) == 0:
			self.reset(observation, goal_direction=goal_direction)
		action = self.current_plan.pop(0) # TODO: should use a deque() instead
		return action
	
	def generate_plan(self, grid, agent_pos, agent_dir, goal_direction=None):
		''' 
		return: plan, goal and sub_goals
		plan: sequence of Actions 
		goal: the final goal position (x,y, Directions) in the plan
		sub_goals: sequence of sub_goals along the plan including the start
		'''
		goals = self.generate_goals(grid,  agent_pos, agent_dir)

		# attempt to find a valid plan for each goal
		for goal in goals:
			plan, sub_goals = a_star_planner(
				grid=grid,
				valid_actions=self.valid_actions,
				init_direction=agent_dir,
				start_pos=agent_pos,
				goal_pos=goal,
				colliders=self.colliders_int,
				goal_direction=goal_direction,
			)
			if plan:  # return the first valid plan found
				self.set_goal_found(True)
				return plan, goal, sub_goals

		# no valid plan found, randomly turn twice in place (so we return goal as agent_pos)
		# this is used to make sure the agent continues exploring
		# but if we want to know if a plan was not found we set the variable goal_not_found
		self.set_goal_found(False)

		action_choices = [Actions.left, Actions.right]
		action_random = random.choice(action_choices)
		return [action_random]*2, agent_pos, [(agent_pos[0],agent_pos[1], agent_dir)]
	
	def generate_goals(self, grid, agent_pos, agent_dir):
		# all potential goals
		goals = [
			(row, col, dir)
			for dir in Directions
			for col in range(grid.shape[1])
			for row in range(grid.shape[0])
			if grid[row, col] not in self.colliders_int
		]
		
		# sort goals by furthest distance from the start position
		goals.sort(
			key=lambda g: heuristic(
				SearchConfiguration(g[0], g[1], g[2]),
				SearchConfiguration(agent_pos[0], agent_pos[1], agent_dir)
			),reverse=True)
		
		goal_visible, goal_pos = self.goal_visible(grid)
		if (self.mode == 'global'):
			goals = [goal_pos] # in global mode we only cosider paths to the goal
		elif goal_visible: 
			goals.insert(0, goal_pos) # if goal is visible we set as the first to try (highest priority)
		return goals

	def goal_visible(self, grid):
		goal_id = self.goal_entities_int[0] # currently only support one goal entity type
		goal_visible = np.argwhere(grid == goal_id)
		if goal_visible.size>0:
			goal_pos = tuple(goal_visible[0].tolist())
			return (True, goal_pos)
		elif self.mode == 'global':
			# in global mode the goal disapears once the agent reaches it
			# so we set the current goal position as the last known goal
			return (True, self.current_goal) 
		return (False, None)