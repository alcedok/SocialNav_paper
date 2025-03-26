import os
import pickle 
import pandas as pd 
import datetime
import time 


'''
Collection of metrics handling classes
'''

class Metrics:
	def __init__(self):
		self.metrics = {}

	def update(self, model_name, metric_name, value):
		if model_name not in self.metrics:
			self.metrics[model_name] = {}
		if metric_name not in self.metrics[model_name]:
			self.metrics[model_name][metric_name] = []
		
		# if the incoming value is a list collapse them into one list
		if isinstance(value, list):
			self.metrics[model_name][metric_name].extend(value)
		else: 
			self.metrics[model_name][metric_name].append(value)
			
	def get_summary(self):
		raise NotImplementedError
	
	def reset(self, metric_name=None):
		if (metric_name is not None) and (metric_name in self.metrics.keys()):
			del self.metrics[metric_name]
		elif  (metric_name is not None) and (metric_name not in self.metrics.keys()):
			raise KeyError('metric_name \'{}\' not in metric keys'.format(metric_name))
		else:
			self.metrics.clear()

	def __getitem__(self, name):
		if name not in self.metrics:
				raise KeyError('metric \'{}\' does not exist'.format(name))
		return self.metrics[name]

	def save(self, save_fpath):
		try:
			directory = os.path.dirname(save_fpath) # get the directory part of the full path
			os.makedirs(directory, exist_ok=True)
			with open(save_fpath, 'wb') as f:
				pickle.dump(self.metrics, f)
			print('Metrics saved: {}'.format(save_fpath))
		except Exception as e:
			print('Error saving data: {}'.format(e))

	@classmethod
	def load(cls, fpath):
		with open(fpath, 'rb') as f:
			metrics = pickle.load(f)
		instance = cls()
		instance.metrics = metrics
		return instance
	
	def to_df(self):
		data = []
		for model_name, metrics in self.metrics.items():
			for metric_name, values in metrics.items():
				for i, value in enumerate(values):
					data.append({'model_name': model_name, 'metric_name': metric_name, 'value': value, 'episodes': i})
		return pd.DataFrame(data)

class TrainingCallback:
	def __init__(self, metrics, model_name):
		self.metrics = metrics
		self.model_name = model_name

	def __call__(self, metric_name, value):
		self.metrics.update(self.model_name, metric_name, value)

class MetricTracker:
	def __init__(self, experiment_id='', backup_path=None):
		self.metrics = {}
		self.experiment_id = experiment_id
		self.backup_fname = self.generate_filename(experiment_id)
		self.backup_path, self.backup_fpath = self.init_backup_path(backup_path, self.backup_fname)

	def generate_filename(self, experiment_id):
		datetime_ = datetime.datetime.now().strftime("%Y%m%d")
		seconds_ = int(time.time()) 
		fname = '{}_{}-{}'.format(experiment_id, datetime_, seconds_)
		return fname
	
	def init_backup_path(self, backup_path, backup_fname):
		backup_fpath = None
		if backup_path:
			if not os.path.isdir(backup_path):
				os.makedirs(backup_path)
			backup_fpath = os.path.join(backup_path, backup_fname)
			print('MetricTracker backup fpath: {}'.format(backup_fpath))
		return backup_path, backup_fpath

	def track(self, name, value, epoch=None, batch=None):
		if name not in self.metrics:
			self.metrics[name] = []
		self.metrics[name].append((epoch, batch, value))
		self._save_backup()
	
	def get_metric(self, name):
		return self.metrics.get(name, None)
	
	def reset(self):
		self.metrics = {}
	
	def is_empty(self):
		return not self.metrics # same as len(self.metrics) == 0
	
	def get_epoch_recon_accuracy(self, name, epoch, total_count):
		''' Calculate the accuracy of the reconstructions agains the actuals '''
		metric_data = self.get_metric(name)
		if metric_data is None:
			raise ValueError('Metric \'{}\' does not exist.'.format(name))
		
		correct_count = []
		for epoch_i, batch_i, value_i in self.get_metric(name):
			if epoch_i == epoch:
				correct_count.append(value_i)

		assert  len(correct_count) != 0, 'there was not data for metric \'{}\' and epoch \'{}\''.format(name, epoch)

		return 100.0 * (sum(correct_count) / total_count) if total_count > 0 else 0.0
	
	def get_epoch_average(self, name):
		''' Calculate the average of each metric across batches for each epoch '''
		metric_data = self.get_metric(name)
		if metric_data is None:
			raise ValueError('Metric \'{}\' does not exist.'.format(name))
		
		epoch_data = {}
		for epoch, batch, value in self.get_metric(name):
			epoch_data.setdefault(epoch, []).append(value)
		return [sum(vals) / len(vals) for epoch, vals in sorted(epoch_data.items())]

	def get_episode_total_reward(self, name):
		''' get the total reward per episode '''
		metric_data = self.get_metric(name)
		if metric_data is None:
			raise ValueError('Metric \'{}\' does not exist.'.format(name))
		
		episode_data = {}
		for episode, batch, value in metric_data:
			episode_data.setdefault(episode, []).append(value)
		
		return [sum(vals) for episode, vals in sorted(episode_data.items())]

	def _save_backup(self):
		if self.backup_path:
			with open(self.backup_fpath, 'wb') as f:
				pickle.dump(self.metrics, f)
	
	@classmethod
	def load_from_backup(cls, backup_fpath):
		with open(backup_fpath, 'rb') as f:
			metrics = pickle.load(f)
		instance = cls(backup_fpath)
		instance.metrics = metrics
		return instance
	
	@classmethod
	def to_df(cls):
		data = []
		for name, entries in cls.items():
			for epoch, batch, value in entries:
				data.append({'experiment_id': cls.experiment_id, 'name': name, 'epoch': epoch, 'batch': batch, 'value': value})
		return pd.DataFrame(data) 
