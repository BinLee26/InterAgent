# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn

from utils.moe import MoE

class HRLBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            #print('++++++++inputshape',input_shape)
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.enablemoe = params.get('enableMoE', False)
            self.critic_cnn = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            mlp_input_shape = input_shape[0]

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }

            self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
            self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

            mlp_init = self.init_factory.create(**self.initializer)

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            mu_init(self.mu.weight)
            sigma_init(self.sigma)
                    
            self.build_actor_moe(mlp_input_shape,out_size)
            
            return
        
        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            mu, sigma, aux_loss = self.eval_actor(obs)
            value = self.eval_critic(obs)
            norm_mu = torch.tanh(mu)
            
            return norm_mu, sigma, value, states, aux_loss

        def eval_actor(self, obs):
            a_out, aux_loss = self.actor_moe(obs)
            mu = self.mu_act(self.mu(a_out))
            sigma = mu * 0.0 + self.sigma_act(self.sigma)
            #print('++++++++aux loss:', aux_loss)
            return mu, sigma, aux_loss
            
        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value
        
        def build_actor_moe(self, input_shape, out_size):
            self.actor_moe = MoE(
                input_size = input_shape,
                output_size= out_size,
                hidden_size=1024,
                num_experts = 4,
                k = 2
            )
            return 
        
    def build(self, name, **kwargs):
        net = HRLBuilder.Network(self.params, **kwargs)
        return net
    
class DMBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.shared_network = HRLBuilder.Network(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    self.actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)

            return
        

        def is_rnn(self):
            return False

        def forward(self, obs_dict):
            obs_dict1 = obs_dict.copy()
            obs_dict2 = obs_dict.copy()
            #print('++++7',obs_dict['obs'].shape)
            obs_dict1['obs'] = obs_dict['obs'][0::2]
            obs_dict2['obs'] = obs_dict['obs'][1::2]

            mu1, sigma1, value1, states, aux_loss1 = self.shared_network.forward(obs_dict1)
            #print('++++sigma',sigma1)
            #print("check obs_dict1 size:", obs_dict1["obs"].shape)
            norm_mu1 = torch.tanh(mu1)

            mu2, sigma2, value2, states, aux_loss2 = self.shared_network.forward(obs_dict2)
            norm_mu2 = torch.tanh(mu2)

            norm_mu = torch.cat([norm_mu1.unsqueeze(1), norm_mu2.unsqueeze(1)], dim=1).reshape(-1, self.actions_num)
            value = torch.cat([value1.unsqueeze(1), value2.unsqueeze(1)], dim=1).reshape(-1, 1)
            sigma = torch.cat([sigma1.unsqueeze(1), sigma2.unsqueeze(1)], dim=1).reshape(-1, self.actions_num)

            aux_loss = (aux_loss1 + aux_loss2) * 0.5
            if obs_dict['is_train']:
                return norm_mu, sigma, value, states, aux_loss
            else:
                return norm_mu, sigma, value, states

        def eval_critic(self, obs):
            obs1 = obs[0::2]
            obs2 = obs[1::2]
        
            value1 = self.shared_network.eval_critic(obs1)
            value2 = self.shared_network.eval_critic(obs2)
            value = torch.cat([value1.unsqueeze(1), value2.unsqueeze(1)], dim=1).reshape(-1, 1)
            return value
        

        
    def build(self, name, **kwargs):
        net = DMBuilder.Network(self.params, **kwargs)
        return net
    