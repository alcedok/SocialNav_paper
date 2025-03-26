'''
Miscellaneous functions
'''

from dataclasses import asdict

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register

from confs.definitions import WorldModelEnvConfig
from environments.wrappers import (PartiallyObservable, PriviledgedModelBuilder)

def load_env(config: WorldModelEnvConfig, 
			 config_exclusions={'environment'}):

	config_dict = {k: v for k, v in asdict(config).items() if k not in config_exclusions}

	env_config = {
	'name': 'GridWorld-v0',
	'config': (config.environment, config_dict) }

	register(id=env_config['name'], entry_point=config.environment, kwargs=config_dict)
	env = gym.make(env_config['name'], disable_env_checker=True)
	env = TimeLimit(env, config.max_steps)
	
	# setup wrappers here 
	if config.priviledge_mode:	
		env = PriviledgedModelBuilder(env, config.agent_view_size)
	else:
		env = PartiallyObservable(env, config.agent_view_size)
	
	return env