from rl_games.algos_torch import network_builder
import torch
import torch.nn as nn
from learning import dm_network_builder
from learning.dm_network_builder import DMBuilder

class MCPBaseBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    
    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            # self.self_obs_size = kwargs['self_obs_size']
            # self.task_obs_size = kwargs['task_obs_size']
            # self.task_obs_size_detail = kwargs['task_obs_size_detail']
            self.has_softmax = params.get("has_softmax", True)
            self.ending_act = params.get("ending_act", True)

            super().__init__(params, **kwargs)

            # Number of primitive policies to compose
            self.num_primitive = kwargs["mcp_details"].get("num_prim", 3)

            composer_input_size = self._calc_input_size(kwargs['input_shape'], self.actor_cnn)
            # Build composer network for generating weights
            composer_mlp_args = {
                'input_size': composer_input_size,
                'units': self.units + [self.num_primitive],
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }

            self.composer = self._build_mlp(**composer_mlp_args)
            
            # Add softmax layer for weight normalization if specified
            if self.has_softmax:
                print("Adding softmax layer to composer network")
                self.composer.append(nn.Softmax(dim=1))
                
            # Option to remove final activation
            if not self.ending_act:
                self.composer = self.composer[:-1]

            if self.is_continuous:
                if (not self.space_config['learn_sigma']): 
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(self.num_primitive, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs):

            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            a_out = self.composer(a_out)

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits
            
            if self.is_continuous:
                # mu = self.mu_act(self.mu(a_out))
                mu = a_out
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma
            return

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

        def get_prim_size(self):
            return self.num_primitive

    def build(self, name, **kwargs):
        net = MCPBaseBuilder.Network(self.params, **kwargs)
        return net

class MCPNetBuilder(DMBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = MCPNetBuilder.Network(self.params, **kwargs)
        return net

    class Network(DMBuilder.Network):
        def __init__(self, params, **kwargs):

            super().__init__(params, **kwargs)
            self.shared_network = MCPBaseBuilder.Network(params, **kwargs)
            self.actions_num = self.shared_network.get_prim_size()
            return