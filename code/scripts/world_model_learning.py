import os
import argparse 

from helpers import env_utils
from helpers.check_dev_env import check_device
from helpers.metrics_utils import Metrics, TrainingCallback
from helpers.visualization_utils import training_plots
from agents.pathfinding import PathfindingAgent
from agents.switching import SwitchingAgent
from agents.random import RandomAgent
from confs.instances import (env_config, wm_training_config, wm_config)
from models import world_model


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_save_dir', type=str, default='../data/models/', help='root directory path where to save model')
	parser.add_argument('--figures_save_dir', type=str, default='../data/models/', help='root directory path where to save figures')
	return parser.parse_args()

def main():
	'''
	By default artifacts are saved in the following directory structure:
	data/
		models/
			<MODEL-ID>/
				checkpoint_<TIMESTAMP>.pth
				checkpoint_<TIMESTAMP>.figure
	'''
	# check device 
	device = check_device()

	args = parse_arguments()
	model_save_dir = args.model_save_dir
	figures_save_dir = args.figures_save_dir

	# load environment 
	env = env_utils.load_env(env_config)

	# load experience collection policies
	valid_actions = env.unwrapped.valid_actions
	
	random_agent= RandomAgent(valid_actions)
	pathfinding_agent =  PathfindingAgent(valid_actions,  mode='global')
	pathfinding_agent.goal_entities = {'goal'}
	agent_list = [pathfinding_agent, random_agent]

	expert_policy_weight = 1-wm_training_config.random_policy_weight
	sampling_weights = [expert_policy_weight, wm_training_config.random_policy_weight]
	switching_agent = SwitchingAgent(agent_list, sampling_weights)
	
	model_id = 'world-model__rollouts-{}__epochs-{}__batchSize-{}__tempAnneal-{}__initTemp-{}'.format(wm_training_config.warm_up_rollouts,
																							wm_training_config.epochs,
																							wm_training_config.batch_size,
																							wm_training_config.temp_anneal,
																							wm_training_config.initial_temperature)
	# setup training metrics tracking
	metrics = Metrics()
	wm_metric_callback = TrainingCallback(metrics, 'world_model_warm-up')

	# train world model
	wm_training_config.model_id = model_id
	model, model_optimizer = world_model.WorldModel.train(wm_config, 
											wm_training_config, 
											env, 
											switching_agent, 
											wm_metric_callback,
											device=device)
	
	print('Finished training')

	# save model
	model_checkpoint_filename,  _ = world_model.save_model_checkpoint(model=model, optimizer=model_optimizer, epoch=wm_training_config.epochs, root_save_path=model_save_dir)
	print('Model ID:',model_id)

	# generate plots/figures
	print('Generate training plots and results')
	figures_save_fpath = os.path.join(figures_save_dir, model_id, model_checkpoint_filename.split('.')[0]+'.jpg')
	training_plots(metrics['world_model_warm-up'], save_path=figures_save_fpath)
	return 

if __name__ == "__main__":
	main()