import numpy as np
import math 
import random
from typing import Any, Tuple
from environments.constants import Directions
from environments.person_of_interest import POI
from minigrid.core.grid import Grid
from minigrid.core.constants import TILE_PIXELS, OBJECT_TO_IDX, COLORS
from minigrid.core.world_object import WorldObj, Ball
from minigrid.utils.rendering import (downsample,
									fill_coords,
									highlight_img,
									point_in_rect,
									point_in_triangle,
									rotate_fn)

'''
Changes to core Minigrid implementation to allow for modeling additional agent.
'''

class AugmentedGrid(Grid):
	''' 
	Minigrid Grid class augmented to render POI 
	Notes: 
	- Minigrid hardcodes and gives priviledge to how the agent is rendered
	- Minigrid also cleverly caches rendering tiles, but only prioritizes the agent, so POI was not getting updated using the original implementation
	- I had to bypass this for redering POI by inhering the class and modifying a few lines line (documented below)
	- There may be a better way of doing this...
	'''
	def __init__(self, width: int, height: int):
		super().__init__(width=width, height=height)

	@classmethod
	def render_tile(
		cls,
		obj: WorldObj | None,
		agent_dir: int | None = None,
		poi_dir: int | None = None, #NOTE: line added
		highlight: bool = False,
		tile_size: int = TILE_PIXELS,
		subdivs: int = 3,
	) -> np.ndarray:
		"""
		Render a tile and cache the result
		"""

		# Hash map lookup key for the cache
		key: tuple[Any, ...] = (agent_dir, highlight, tile_size, poi_dir)
		key = obj.encode() + key if obj else key
		
		if key in cls.tile_cache:
			return cls.tile_cache[key]

		img = np.zeros(
			shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
		)

		# Draw the grid lines (top and left edges)
		fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
		fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

		if obj is not None:
			obj.render(img)

		# Overlay the agent on top
		if agent_dir is not None:
			tri_fn = point_in_triangle(
				(0.12, 0.19),
				(0.87, 0.50),
				(0.12, 0.81),
			)
			# Rotate the agent based on its direction
			tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
			fill_coords(img, tri_fn, (255, 0, 0))

		# Highlight the cell if needed
		if highlight:
			highlight_img(img)

		# Downsample the image to perform supersampling/anti-aliasing
		img = downsample(img, subdivs)

		# Cache the rendered tile
		cls.tile_cache[key] = img

		return img

	def render(
		self,
		tile_size: int,
		agent_pos: tuple[int, int],
		agent_dir: int | None = None,
		highlight_mask: np.ndarray | None = None,
	) -> np.ndarray:
		"""
		Render this grid at a given scale
		:param r: target renderer object
		:param tile_size: tile size in pixels
		"""

		if highlight_mask is None:
			highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

		# Compute the total grid size
		width_px = self.width * tile_size
		height_px = self.height * tile_size

		img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

		# Render the grid
		for j in range(0, self.height):
			for i in range(0, self.width):
				cell = self.get(i, j)
				poi_here = cell.type=='ball' if cell is not None else False 	#NOTE: line added
				if poi_here:													#NOTE: line added
					poi_dir = cell.get_direction() 								#NOTE: line added
				agent_here = np.array_equal(agent_pos, (i, j))
				assert highlight_mask is not None
				tile_img = AugmentedGrid.render_tile( 							#NOTE:line that was changed
					cell,
					agent_dir=agent_dir if agent_here else None,
					poi_dir=poi_dir if poi_here else None,						#NOTE: line added
					highlight=highlight_mask[i, j],
					tile_size=tile_size,
				)

				ymin = j * tile_size
				ymax = (j + 1) * tile_size
				xmin = i * tile_size
				xmax = (i + 1) * tile_size
				img[ymin:ymax, xmin:xmax, :] = tile_img

		return img
	
	@staticmethod
	def decode(array: np.ndarray) -> tuple[Grid, np.ndarray]:
		"""
		Decode an array grid encoding back into a grid
		"""

		width, height, channels = array.shape
		assert channels == 3

		vis_mask = np.ones(shape=(width, height), dtype=bool)

		grid = AugmentedGrid(width, height)
		for i in range(width):
			for j in range(height):
				type_idx, color_idx, state = array[i, j]
				v = WorldObj.decode(type_idx, color_idx, state)
				if v is not None and v.type == 'ball':  					#NOTE: line added
					v = POI()												#NOTE: line added
					v.set_direction(Directions(state))						#NOTE: line added

				grid.set(i, j, v)
				vis_mask[i, j] = type_idx != OBJECT_TO_IDX["unseen"]

		return grid, vis_mask