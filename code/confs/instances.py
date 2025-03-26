
# hide pygame message and gymnasium warnings
import os
import warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
# warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')
from minigrid.core.constants import OBJECT_TO_IDX
from environments.crossing import CrossingEnv
from confs.definitions import (WorldModelEnvConfig, 
                            WorldModelTrainingConfig, 
                            WorldModelConfig,
                            WorldModelCheckpointConfig)

## --------------------------------
## World Model
env_config = WorldModelEnvConfig(
                                 environment=CrossingEnv,
                                 num_crossings=1,
                                 width=9, height=9, 
                                 agent_view_size=5, 
                                 max_steps=200,
                                 see_through_walls=True,
                                 priviledge_mode=True)

wm_training_config = WorldModelTrainingConfig(warm_up_rollouts=3000, 
											  epochs=20, batch_size=500, 
											  temp_anneal=False, 
											  initial_temperature=0.6)

# use defaults unless specified
wm_config = WorldModelConfig()

# Saved model configuration
wm_checkpoint_config = WorldModelCheckpointConfig(
    model_id_chpt='world-model__rollouts-3000__epochs-20__batchSize-500__tempAnneal-False__initTemp-0.6',
    checkpoint_file='checkpoint_20250321-1742588693.pth',
    epoch=20
)

## --------------------------------
## Reinforcement Learning
# ...
# because configs are defined given inputs from the training script
# they are instantiated in `scripts/policy_learning_experiments.py`
