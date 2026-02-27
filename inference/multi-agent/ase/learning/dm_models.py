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

import torch
import torch.nn as nn
from learning import amp_models
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

class ModelDMContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return
    
    def build(self, config):
        #config['input_shape'] = (2 * config['input_shape'][0], )
        #print('++++network_config',config['input_shape'])
        net = self.network_builder.build('dual', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelDMContinuous.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return
        
        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            is_moe = input_dict.get('is_moe', False)
            if is_moe and is_train:
                mu, logstd, value, states, aux_loss = self.a2c_network(input_dict)
            else:
                mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                if is_moe:
                    result.update({'aux_loss': aux_loss})                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                if is_moe:
                    result.update({'aux_loss': aux_loss})
                return result
            
        # def forward(self, input_dict):
        #     is_train = input_dict.get('is_train', True)
        #     result = super().forward(input_dict)
            
            return result