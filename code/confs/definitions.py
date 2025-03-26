
from dataclasses import dataclass, field 
from typing import Set, Literal, Optional

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.world_object import WorldObj, Wall

# Environment Configuration
valid_actions = {Actions.left, Actions.right, Actions.forward, Actions.pickup}

# object index remapping for reducing the number of concept dimensions in the decoder
OBJECT_IDX_REMAP = {
                OBJECT_TO_IDX['empty'] :  0, 
                OBJECT_TO_IDX['wall']  :  1,
                OBJECT_TO_IDX['goal']  :  2,
                # OBJECT_TO_IDX['ball']  :  3, # we assume upstream process provides info about POI, so we don't model it
                }

REVERSED_OBJECT_IDX_REMAP = {v:k for k,v in OBJECT_IDX_REMAP.items()}

@dataclass
class WorldModelEnvConfig:
    environment: MiniGridEnv = None
    valid_actions: Set[str] = field(default_factory = lambda: valid_actions)
    width: int = 11
    height: int = 11
    num_crossings: int = 1
    highlight: bool = True
    max_steps: int = 200
    agent_view_size: int = 5
    see_through_walls: bool = False
    priviledge_mode: bool = False # only used when collecting experiences using expert planner 
    render_mode: str = 'rgb_array' # only tested on this mode
    obstacle_type: WorldObj = Wall # only tested on this object

# Training Configuration
@dataclass
class WorldModelTrainingConfig:
    model_id: str = '00'
    random_policy_weight: float = 0.5 # number between [0-1]
    warm_up_rollouts: int = 200
    max_steps: int = field(default = WorldModelEnvConfig().max_steps)

    epochs: int = 10
    batch_size: int = 10

    initial_learning_rate: float = 1e-2
    grad_clip_norm: float = 100
    learning_rate_gamma: float = 0.9

    temp_anneal: bool = False
    initial_temperature: float = 0.6
    minimum_temperature: float = 0.6
    temperature_anneal_rate: float = 0.05
    
    kl_loss_weight: float = 0.1
    
    compute_proposed_class_weights: bool = True

# Observation Model Configuration
@dataclass
class ObservationModelConfig:
    categorical_dim: int
    num_categorical_distributions: int
    gumbel_temperature: float
    concept_embed_dim: int
    action_embed_dim: int
    prior_mode: Literal['uniform', 'prior_net']
    object_idx_lookup: dict
    object_idx_rlookup: dict 
    observation_dim: int = field(default = WorldModelEnvConfig().agent_view_size)
    num_att_heads: int = 1
    conv1_hidden_dim: int = 4
    attention_belief_model = None   # defined in World Model constructor
    belief_action_fc = None         # defined in World Model constructor
    @property
    def concept_dim(self):
        return len(self.object_idx_lookup.keys())
    
# Transition Model Configuration
@dataclass
class TransitionModelConfig:
    categorical_dim: int
    num_categorical_distributions: int
    gumbel_temperature: float
    action_embed_dim: int
    num_actions: int = field(default_factory = lambda: len(WorldModelEnvConfig().valid_actions))
    fc1_hidden_dim: int = 200 #200?
    use_belief_action_cross_attn: bool = False
    attention_belief_model = None   # defined in World Model constructor
    belief_action_fc = None         # defined in World Model constructor

# Action Model Configuration
@dataclass
class ActionModelConfig:
    action_embed_dim: int
    num_actions: int = field(default_factory = lambda: len(WorldModelEnvConfig().valid_actions))

# Inverse Model Configuration
@dataclass
class InverseModelConfig:
    categorical_dim: int
    num_categorical_distributions: int
    gumbel_temperature: float
    action_embed_dim: int
    num_actions: int = field(default_factory = lambda: len(WorldModelEnvConfig().valid_actions))
    fc1_hidden_dim: int = 100

# World Model Configuration
@dataclass
class WorldModelConfig:
    object_idx_lookup: dict = field(default_factory=lambda: OBJECT_IDX_REMAP.copy())
    object_idx_rlookup: dict = field(default_factory=lambda: REVERSED_OBJECT_IDX_REMAP.copy())
    concept_dim: int = field(default_factory = lambda: len(WorldModelConfig().object_idx_lookup.keys())) #len(object_idx_lookup.keys())
    categorical_dim: int = 8
    num_categorical_distributions: int = 8
    observation_dim: int = field(default = WorldModelEnvConfig().agent_view_size)
    gumbel_hard: bool = False
    gumbel_temperature: float = field(default = WorldModelTrainingConfig().initial_temperature)
    num_actions: int = field(default_factory = lambda: len(WorldModelEnvConfig().valid_actions))
    action_embed_dim: int = 10 # dependent on categorical_dim because it is required for attention layers to match matrix multiplication
    concept_dim: int =  0 # computed from the number of keys in object_idx_lookup 
    use_inverse_model: bool = True # NOTE: disabling not implemented
    prior_mode: Literal['uniform', 'prior_net'] = 'uniform'

    @property
    def observation_model_config(self):
        return ObservationModelConfig(
                self.categorical_dim, 
                self.num_categorical_distributions, 
                self.gumbel_temperature, 
                self.observation_dim,
                self.action_embed_dim,
                self.prior_mode,
                self.object_idx_lookup,
                self.object_idx_rlookup)
    @property
    def action_model_config(self):
        return ActionModelConfig(
                self.action_embed_dim)
    @property
    def transition_model_config(self):
        return TransitionModelConfig(
                self.categorical_dim, 
                self.num_categorical_distributions,
                self.gumbel_temperature, 
                self.action_embed_dim)
    @property
    def inverse_model_config(self):
        return InverseModelConfig(
            self.categorical_dim, 
            self.num_categorical_distributions,
            self.gumbel_temperature, 
            self.action_embed_dim)
    
@dataclass
class BeliefDQNModelTraining:
    influence_mode: Literal['zero', 'random', 'perfect-information', 'perspective-shift']
    model_id: str = '00'
    experiment_name: str = ''
    num_episodes: int = 10000
    batch_size: int = 64

    memory_size: int = 60000 #int(1e6)

    goal_completion_window: int = 100

    grad_clip_norm: float = 10
    gamma: float = 0.90
    #-------------------------------
    # enabled after warmpup period
    eps_start: float = 0.7
    eps_end: float = 0.01
    eps_decay_coeff: float = 0.2 # in the training code: eps_decay=num_episodes*eps_decay_coeff,  when episode==eps_decay prob drop to ~36% of original value
    #-------------------------------
    # enabled when using soft update
    tau: float = 0.005 
    #-------------------------------
    learning_rate: float = 1e-3
    target_update_freq: int = 1000
    target_update_mode: Literal['hard', 'soft'] = 'hard'
    model_update_freq: int = 1 # steps post warm-up
    warmup: bool = True
    warmup_steps: int = 1000

    save_video_episode_freq: int = 1000
    video_path: str = 'videos/dqn/training/'
    video_name_prefix: str = ''
    
    timestamp: int = 0 # used as the top dir name in logs and video

valid_actions_policy = valid_actions

@dataclass
class BeliefDQNModelConfig:
    categorical_dim: int = field(default = WorldModelConfig().categorical_dim)
    num_categorical_distributions: int = field(default = WorldModelConfig().num_categorical_distributions)
    valid_actions: Set[str] = field(default_factory = lambda: valid_actions_policy)
    num_actions: int = field(default_factory = lambda: len(valid_actions_policy))
    action_embed_dim: int = field(default = WorldModelConfig().action_embed_dim)
    observation_dim: int = field(default = WorldModelEnvConfig().agent_view_size)
    see_through_walls: bool = field(default = WorldModelEnvConfig().see_through_walls)
    hidden_dim: int = 500
    num_influences: int = 1 # in our experiments we only explore 1 influence point, so the # of inputs to DQN become (self_belief + num_influences)
    goal_entities: str = 'ball' # only one single goal entity is supported at a time, ball or goal

# policy loading config
@dataclass
class WorldModelCheckpointConfig:
    model_id_chpt: str 
    checkpoint_file: str 
    epoch: int
