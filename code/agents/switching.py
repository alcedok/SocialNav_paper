import random 

from agents.abstract import Agent

class SwitchingAgent(Agent):
    '''
    Meta-agent that switches policy randomly based on param: sampling_weights
    
    agents in agent_list must be of type Agent to ensure compatability downstream
    '''
    def __init__(self, 
                 agent_list, 
                 sampling_weights,  
                 agent_name='switching-agent',
                 verbose=False):
        
        self.agent_list = agent_list
        self.sampling_weights = sampling_weights
        self.agent_selected = self.switch_agent()
        self.agent_name=agent_name
        self.verbose = verbose
    
    def switch_agent(self):
        return random.choices(self.agent_list, weights=self.sampling_weights, k=1)[0]
    
    def reset(self, observations=None):
        for agent in self.agent_list:
             agent.reset(observations)
        self.agent_selected = self.switch_agent()
        if self.verbose:
            print('Using: {}'.format(self.agent_selected.agent_name))
        
    def act(self, observations=None):
        return self.agent_selected.act(observations)