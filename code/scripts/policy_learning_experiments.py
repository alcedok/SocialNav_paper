import os
import sys
import argparse 
import logging
from dataclasses import dataclass, asdict

from confs.definitions import BeliefDQNModelConfig, BeliefDQNModelTraining
from confs.instances import (env_config, wm_training_config, wm_checkpoint_config)
from helpers.env_utils import load_env
from helpers.check_dev_env import check_device
from helpers.metrics_utils import Metrics, TrainingCallback
from models.world_model import load_model_from_checkpoint as load_wm_checkpoint
from models.belief_dqn import BeliefDQNModel
from environments.crossing import CrossingEnv
from environments.person_following import PersonFollowingEnv

from gymnasium.wrappers import RecordVideo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment_type', type=str, required=True, help='single-agent or multi-agent')
    parser.add_argument('--influence_mode', type=str, required=True, help='Influence-mode, one of: {zero | random | perfect-information | perspective-shift}')
    parser.add_argument('--timestamp', type=str, required=True, help='timestamp of the experiment')
    parser.add_argument('--instance_id', type=str, required=True, help='unique ID for this training instance')
    parser.add_argument('--logfile', type=str, required=True, help='path to logfile to append to')
    return parser.parse_args()

def load_world_model(model_checkpoint_config, device, frozen=True):
    world_model, world_model_optimizer, wm_epoch_chpt = load_wm_checkpoint(wm_training_config, 
                                                                        model_id=model_checkpoint_config.model_id_chpt, 
                                                                        checkpoint_file=model_checkpoint_config.checkpoint_file, 
                                                                        epoch=model_checkpoint_config.epoch, 
                                                                        frozen=frozen,
                                                                        device=device)
    return world_model, world_model_optimizer, wm_epoch_chpt

def setup_env(experiment_name, training_config, wm_env_config):
    env = load_env(wm_env_config)
    run_env = RecordVideo(env, 
                        video_folder=training_config.video_path, 
                        name_prefix=experiment_name, disable_logger=True,
                        episode_trigger=lambda x: (x % training_config.save_video_episode_freq == 0) or 
                                                    (x == training_config.num_episodes-1))
    return run_env 

def main():
    args = parse_arguments()
    
    environment_type = args.environment_type
    influence_mode = args.influence_mode
    experiment_name = influence_mode
    timestamp = args.timestamp
    instance_id = args.instance_id

    all_environment_types = {'multi-agent'}
    all_influence_modes = {'zero', 'random','perfect-information', 'perspective-shift'}
    assert environment_type in all_environment_types, 'environment_type [{}] not implemented. Check spelling.'.format(environment_type)
    assert influence_mode in all_influence_modes, 'influence_mode [{}] not implemented. Check spelling.'.format(influence_mode)

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

    device = check_device()

    if environment_type == 'multi-agent':
        environment_class = PersonFollowingEnv
        model_checkpoint_config = wm_checkpoint_config
    else:
        return NotImplementedError('environment_type {} not supported'.format(environment_type))

    experiment_name = '{}__{}__{}__{}'.format(environment_type, experiment_name, timestamp, instance_id)
    video_path =  os.path.join( 'videos/dqn/training/',timestamp+'/',experiment_name+'/')
    goal_entities = 'ball' if environment_type=='multi-agent' else 'goal'

    # define all configurations
    env_config.environment = environment_class
    print('---')
    print(env_config.environment)
    print(env_config)
    print('---')
    policy_config = BeliefDQNModelConfig(goal_entities=goal_entities)

    training_config = BeliefDQNModelTraining(influence_mode=influence_mode,
                                             experiment_name=experiment_name,
                                             video_path=video_path,
                                             timestamp=timestamp)

    # include configs in log for documentation
    logging.log(logging.INFO, asdict(model_checkpoint_config))
    logging.log(logging.INFO, asdict(env_config))
    logging.log(logging.INFO, asdict(policy_config))
    logging.log(logging.INFO, asdict(training_config))

    # load environment 
    env = setup_env(experiment_name, training_config, env_config)

    # load world model 
    world_model, _, _ = load_world_model(model_checkpoint_config, device, frozen=True)

    # init metrics 
    metrics = Metrics()
    policy_learning_metric_callback = TrainingCallback(metrics, influence_mode)

    # train policy
    policy, policy_optimizer = BeliefDQNModel.train(
    										world_model,
											policy_config, 
											training_config, 
											env, 
											policy_learning_metric_callback,
                                            device=device)

    # save the metrics for analysis
    metrics_fpath = 'data/{}/{}.pkl'.format(influence_mode, experiment_name)
    metrics.save(metrics_fpath)

    return 

if __name__ == "__main__":
    main()