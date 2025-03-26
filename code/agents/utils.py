from heapq import heappop, heappush
import numpy as np 
import random
from collections import deque
from typing import NamedTuple, Optional

from minigrid.core.actions import Actions

from environments.constants import Directions, DirectionsXY

class SearchConfiguration(NamedTuple):
    x: int 
    y: int
    direction: Optional[Directions] = None 

def heuristic(a: SearchConfiguration, b: SearchConfiguration):
    '''
    Manhattan distance with (optional) orientation penalty
    '''
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    
    if (a.direction is not None) and (b.direction is not None):
        # calculate the minimum angle difference in orientation
        # directions are integers [0-3], see Directions 
        a_dir = a.direction.value
        goal_dir = b.direction.value
        do = min(abs(a_dir - goal_dir), 4 - abs(a_dir - goal_dir))
        orientation_penalty = 0.5
        return dx + dy + orientation_penalty * do
    else:
        return dx + dy

def reconstruct_path(came_from, current):
    path = [] # sequence of Actions Enum
    sub_goals = [(current.x, current.y, current.direction)] # sequence of (x,y, Directions) points
    while current in came_from:
        current, action = came_from[current]
        path.append(action)
        sub_goals.append((current.x, current.y, current.direction))
    path.reverse()
    sub_goals.reverse()
    return path, sub_goals

def a_star_planner(grid: np.array, 
                   valid_actions: set,
                   init_direction: Directions, 
                   start_pos: tuple, 
                   goal_pos: tuple, 
                   colliders: set, 
                   goal_direction: Directions = None):
    '''
     A* search algorithm for 2D grid navigation with optional goal orientation

    input:
        valid_actions: set of actions allowed, Actions from minigrid.core.constants 
        grid: A 2D grid representing the environment
        init_direction: The initial direction of the agent
        start_pos: start position in the grid
        goal_pos: goal position in the grid
        goal_orientation: The desired goal orientation (optional has to be an Enum Directions)

    output:
        list of actions representing the path to the goal, or None if no path exists
    '''
    start_state = SearchConfiguration(x=start_pos[0], y=start_pos[1], direction=init_direction)
    goal_state = SearchConfiguration(x=goal_pos[0], y=goal_pos[1], direction=goal_direction)

    frontier = []
    heappush(frontier, (0, start_state))
    came_from = {}
    cost_so_far = {start_state: 0}
    
    while frontier:
        _, current = heappop(frontier)

        # check if the current state config matches the goal, including orientation if specified
        if current.x == goal_state.x and current.y == goal_state.y:
            if not goal_state.direction or current.direction == goal_state.direction:
                return reconstruct_path(came_from, current)
        
        # explore neighbors using actions
        for action in valid_actions:
            if action == Actions.forward:
                dx, dy = DirectionsXY[current.direction.name].value
                next_x, next_y = current.x + dx, current.y + dy
                next_dir = Directions(current.direction)
            elif action == Actions.left:
                next_x, next_y = current.x, current.y
                next_dir = Directions((current.direction - 1) % len(Directions))
            elif action == Actions.right:
                next_x, next_y = current.x, current.y
                next_dir = Directions((current.direction + 1) % len(Directions))
            next_state = SearchConfiguration(next_x, next_y, next_dir)
             # validate move
            if not (0 <= next_state.x < grid.shape[0] and 0 <= next_state.y < grid.shape[1]):
                continue
            if grid[next_state.x, next_state.y] in colliders:
                continue

            # calculate the new cost
            new_cost = cost_so_far[current] + 1  # could add variable costs here
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state, goal_state)
                heappush(frontier, (priority, next_state))
                came_from[next_state] = (current, action)
                
    return None, None # no feasible plan or sub_goals

def is_path_within_distance(grid: np.array, 
                            start_pos: tuple, 
                            goal_pos: tuple, 
                            colliders: set, 
                            max_distance: int) -> bool:
    '''
    Check if there exists a path from start_pos to goal_pos with a path length 
    (number of moves) less than or equal to max_distance, accounting for obstacles
    
    input:
        grid (np.array): A 2D grid representing the environment
        start_pos (tuple): (x, y) coordinates of the starting point
        goal_pos (tuple): (x, y) coordinates of the goal point
        colliders (set): Set of grid values that are considered obstacles
        max_distance (int): Maximum allowed number of steps (Manhattan distance)

    output:
        True if a valid path exists with cost <= max_distance, otherwise False
    '''
    frontier = deque()
    frontier.append((start_pos, 0))  # each element is ((x, y), cost)
    visited = {start_pos}
    
    while frontier:
        (x, y), cost = frontier.popleft()
        
        # vound the goal!
        if (x, y) == goal_pos:
            return True
        
        # stop expanding if we've reached the maximum allowed cost.
        if cost == max_distance:
            continue
        
        # explore the 4-connected neighbors (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            
            # check bounds
            if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                continue
            
            # check if the cell is an obstacle.
            if grid[nx, ny] in colliders:
                continue
            
            # check if already visited.
            if (nx, ny) in visited:
                continue
            
            visited.add((nx, ny))
            frontier.append(((nx, ny), cost + 1))
    
    # If the search completes without finding the goal within max_distance.
    return False
