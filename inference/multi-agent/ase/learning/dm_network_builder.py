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

from learning import ase_network_builder

class HRLBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    #print("======", actions_num)
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
            
            print('+++cnn',self.has_cnn)

            return
        
        def forward(self, obs_dict):
            mu, sigma, value, states = super().forward(obs_dict)
            #print('++++6', obs_dict['obs'].shape)
            norm_mu = torch.tanh(mu)
            return norm_mu, sigma, value, states

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

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
            print("kwargs", kwargs)
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

            mu1, sigma1, value1, states = self.shared_network.forward(obs_dict1)
            #print("check obs_dict1 size:", obs_dict1["obs"].shape)
            norm_mu1 = torch.tanh(mu1)

            mu2, sigma2, value2, states = self.shared_network.forward(obs_dict2)
            norm_mu2 = torch.tanh(mu2)

            norm_mu = torch.cat([norm_mu1.unsqueeze(1), norm_mu2.unsqueeze(1)], dim=1).reshape(-1, self.actions_num)
            value = torch.cat([value1.unsqueeze(1), value2.unsqueeze(1)], dim=1).reshape(-1, 1)
            sigma = torch.cat([sigma1.unsqueeze(1), sigma2.unsqueeze(1)], dim=1).reshape(-1, self.actions_num)

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