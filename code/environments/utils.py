
from typing import Tuple, List
import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.actions import Actions

from environments.constants import Directions, DirectionsXY

def gen_obs_at_point(cur_pos: Tuple[int, int], 
				cur_dir: int, 
				grid: Grid, 
				agent_view_size: int,
				see_through_walls:bool):
	'''
	Modified from minigrid.minigrid_env

	Generate the sub-grid observed by the agent.
	This method also outputs a visibility mask telling us which grid
	cells the agent can actually see.
	
	'''
	topX, topY, botX, botY = get_view_exts(cur_pos, cur_dir, agent_view_size)

	sub_grid = grid.slice(topX, topY, agent_view_size, agent_view_size)

	for i in range(cur_dir + 1):
		sub_grid = sub_grid.rotate_left()

	ego_agent_pos = sub_grid.width // 2, sub_grid.height - 1

	# Process occluders and visibility
	# Note that this incurs some performance cost
	if not see_through_walls:
		vis_mask = sub_grid.process_vis(
			agent_pos = (agent_view_size // 2, agent_view_size - 1)
		)
	else:
		vis_mask = np.ones(shape=(sub_grid.width, sub_grid.height), dtype=bool)

	# NOTE: 
	#  in the original MiniGridEnv.gen_obs_grid they edit the cell at the agent's position
	#  such that if the agent is carrying an object it is placed at that cell
	#  since we don't consider objects being carried in this work
	#  we default to always keeping that cell as None
	sub_grid.set(*ego_agent_pos, None)

	return sub_grid, vis_mask

def get_view_exts(cur_pos: Tuple[int, int], 
				cur_dir: int,
				agent_view_size:int):
	'''
	Modified from minigrid.minigrid_env

	Get the extents of the square set of tiles visible to the agent
	Note: the bottom extent indices are not included in the set
	'''
	# Facing right
	if cur_dir == 0:
		topX = cur_pos[0]
		topY = cur_pos[1] - agent_view_size // 2
	# Facing down
	elif cur_dir == 1:
		topX = cur_pos[0] - agent_view_size // 2
		topY = cur_pos[1]
	# Facing left
	elif cur_dir == 2:
		topX = cur_pos[0] - agent_view_size + 1
		topY = cur_pos[1] - agent_view_size // 2
	# Facing up
	elif cur_dir == 3:
		topX = cur_pos[0] - agent_view_size // 2
		topY = cur_pos[1] - agent_view_size + 1
	else:
		assert False, "invalid agent direction"

	botX = topX + agent_view_size
	botY = topY + agent_view_size

	return topX, topY, botX, botY