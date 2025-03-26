import os 
import glob 
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd 
import random 

from minigrid.wrappers import ReseedWrapper

from confs.instances import (env_config, wm_training_config, wm_checkpoint_config)
from helpers.metrics_utils import Metrics
from helpers.env_utils import load_env
from helpers.check_dev_env import check_device
from helpers.visualization_utils import plot_mental_simulation
from models.world_model import load_model_from_checkpoint as load_wm_checkpoint
from models.utils import mental_vs_real
from agents.pathfinding import PathfindingAgent
from environments.person_following import PersonFollowingEnv

'''
Representation Learning results
'''

def render_perspective_shift_results(root_save_path='data/world-model_learning/'):
	device = check_device()

	# load environment 
	env_config.environment = PersonFollowingEnv
	env = load_env(env_config)
	valid_actions = env.unwrapped.valid_actions
	world_model, _, _ = load_wm_checkpoint(wm_training_config, 
											model_id=wm_checkpoint_config.model_id_chpt, 
											checkpoint_file=wm_checkpoint_config.checkpoint_file, 
											epoch=wm_checkpoint_config.epoch, 
											frozen=True,
											device=device)
	

	# mental simulation from loaded checkpoint
	_seed = random.randint(0,100)
	print('seed:',_seed)

	mental_sim_policy = PathfindingAgent(valid_actions,  mode='local', agent_view_size= env_config.agent_view_size)
	mental_sim_policy.goal_entities = {'ball'}
	mental_sim_policy.observation_grid_key = 'partial_observation_with_poi'

	env_with_seed = ReseedWrapper(env, seeds=(_seed,), seed_idx=0)

	mental_sim_outputs = mental_vs_real(env_with_seed, world_model, mental_sim_policy, 
									 device=device, use_poi_dir_goal=True,
									 root_save_path=root_save_path)
	
	plot_mental_simulation(env_with_seed, mental_sim_outputs)
	save_path = os.path.join(root_save_path,wm_checkpoint_config.model_id_chpt)
	os.makedirs(save_path, exist_ok=True)
	save_path = os.path.join(save_path,'perspective-shift.jpg')
	plt.savefig(save_path, bbox_inches='tight')
	print('Saved perspective-shift figures in directory:\n\t', save_path)
	return 

'''
Metrics handling functions
'''

def get_metrics_path(log_file_path):
	''' get saved metrics files after policy training '''
	try:
		with open(log_file_path, 'r') as f:
			lines = f.readlines()  # Read all lines into a list
			for line in reversed(lines):  # Iterate through the lines in reverse
				if line.startswith('Metrics saved: '):
					return line.split('Metrics saved: ')[1].strip()
			return None  # Return None if not found

	except FileNotFoundError:
		print(f"Error: Log file not found at {log_file_path}")
		return None
	except Exception as e:
		print(f"An error occurred: {e}")
		return None

def get_metrics_df(metrics_fpaths, metric='cumulative_avg_episode_rewards'):
	''' load metrics files into dataframe for plotting '''
	metrics = Metrics()
	dfs = []
	for env_type, influence_mode, exp_id, timestamp, metric_data_fpath in metrics_fpaths:
		if metric_data_fpath:
			metrics_from_file = metrics.load(metric_data_fpath)
			metrics_df = metrics_from_file.to_df()
			metrics_df['environment'] = env_type
			metrics_df['influence_mode'] = influence_mode
			metrics_df['exp_id'] = exp_id
			dfs.append(metrics_df)

	df = pd.concat(dfs)
	print('unique values in column {}: {}'.format(metric, df['metric_name'].unique()))
	df_filtered = df[df['metric_name']==metric]
	metrics_df_melted = pd.melt(df_filtered, 
						id_vars=['environment','influence_mode', 'episodes', 'exp_id'],  # Columns to keep as identifiers
						value_vars=['value'],    # Columns to melt
						var_name='metric_name',  # Name of the new metric name column
						value_name=metric)       # Name of the new value column
	return metrics_df_melted

def plot_policy_metrics(logs_path='logs/', root_save_path='data/policy_learning/', from_backup=None):
	'''
	- collect all policy learning metric log files
	- parse thru them to grab their respective data pkl
	- 
	'''
	#TODO: need to clean this and make better comments 

	def plot(metrics_df_melted):
		# plot and save to file
		fig, ax = plt.subplots(figsize=(8, 6))
		sns.lineplot(data=metrics_df_melted, 
				x='episodes', y='cumulative_avg_episode_rewards', 
				hue='influence_mode',
				err_style="band", estimator='median', 
				errorbar=('ci', 95), n_boot=100, seed=10,
				ax=ax )
		ax.set_title('Person Following Task')
		ax.set_xlabel('Episodes')
		ax.set_ylabel('Avg. Cumulative Episodic Rewards')
		ax.grid(alpha=0.3)

		# adjust legend
		handles, labels = ax.get_legend_handles_labels() 
		order = [2, 0, 1] # re-order the legend
		ax.legend([handles[i] for i in order], [labels[i] for i in order], title='Influence Type', loc='best') 
		
	if from_backup is not None:
		metrics_df_melted_new = pd.read_pickle(from_backup)
		print(metrics_df_melted_new.shape)
		return plot(metrics_df_melted_new)
	
	pattern = os.path.join(logs_path, '*_exp-*')  # create the glob pattern
	all_logs = glob.glob(pattern)
	all_logs = [os.path.basename(f) for f in all_logs] # extract just the file names

	print('All training logs...')
	for l in all_logs:
		print(l)

	metrics_fpaths = []
	env_type_list = []
	influence_mode_list = []
	exp_instance_list = []
	timestamp_list = []
	
	# parse thru log files and collect data file paths
	print('\nAll metrics data for each run...')
	for log in all_logs:
		metric_data_fpath = get_metrics_path(os.path.join(logs_path,log))

		name_parts = log.split('_')
		env_type = name_parts[0]
		influence_mode = name_parts[1]
		exp_id_timstamp = name_parts[2].split('.')[0].split('-')
		exp_id = exp_id_timstamp[1]
		timestamp = exp_id_timstamp[2]
		
		env_type_list.append(env_type)
		influence_mode_list.append(influence_mode)
		exp_instance_list.append(exp_id)
		timestamp_list.append(timestamp)

		metrics_data_info = (env_type, influence_mode, exp_id, timestamp, metric_data_fpath)
		metrics_fpaths.append(metrics_data_info)
		print(metrics_data_info)

	# aggregate data into dataframe
	metrics_df_melted = get_metrics_df(metrics_fpaths, metric='cumulative_avg_episode_rewards')
	
	# plot
	print('Computing aggregate statistics (confidence intervals)')
	print('\tthis takes some time (2-3 mins)...')
	plot(metrics_df_melted)
	
	os.makedirs(root_save_path, exist_ok=True)
	save_path = os.path.join(root_save_path,timestamp+'.jpg')
	plt.savefig(save_path)
	print('Saved policy learning plots in directory:\n\t', save_path)
	
	# save pkl file for backup
	pkl_fname = os.path.join(root_save_path,timestamp+'.pkl')
	metrics_df_melted.to_pickle(pkl_fname)
	print('Saved dataframe for generating learning plots in directory:\n\t', pkl_fname)

	return