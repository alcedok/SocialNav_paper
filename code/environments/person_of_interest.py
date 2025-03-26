import math 
from typing import Tuple 

from environments.constants import Directions

from minigrid.core.world_object import Ball
from minigrid.core.constants import COLORS
from minigrid.utils.rendering import (fill_coords,
									point_in_triangle,
									rotate_fn)
class POI(Ball):
	''' modeled as a ball since minigrid doens't support multi-agent'''
	def __init__(self, color='blue'):
		super().__init__(color)
		self.direction = None
		self.position = None
	def can_pickup(self):
		return False
	def can_overlap(self) -> bool:
		return False
	def get_direction(self) -> Directions:
		return self.direction
	def set_direction(self, direction: Directions):
		self.direction = direction
	def set_position(self, position: Tuple[int,int]):
		self.position = position
	def get_position(self) ->  Tuple[int,int]:
		return self.position
	def render(self, img):
		tri_fn = point_in_triangle(
				(0.12, 0.19),
				(0.87, 0.50),
				(0.12, 0.81),
			)
		# Rotate the triangle based on its direction
		tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.get_direction())
		fill_coords(img, tri_fn, COLORS[self.color])