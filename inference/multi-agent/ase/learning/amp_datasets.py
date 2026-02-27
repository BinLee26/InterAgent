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
from rl_games.common import datasets

class AMPDataset(datasets.PPODataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):
        super().__init__(batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len)
        self._idx_buf = torch.randperm(batch_size//2)
        self.length = self.length // 2
        #print("@@@@@@@@@", self.length, self.minibatch_size)
        return
    
    def update_mu_sigma(self, mu, sigma):	  
        raise NotImplementedError()
        return

    # def _get_item(self, idx):
    #     start = idx * self.minibatch_size
    #     end = (idx + 1) * self.minibatch_size
    #     sample_idx = self._idx_buf[start:end]

    #     #print("minibatch size:", self.minibatch_size)
    #     #print("sample_idx:", sample_idx)

    #     input_dict = {}
    #     for k,v in self.values_dict.items():
    #         if k not in self.special_names and v is not None:
    #             #print('++++++', k, v.shape)
    #             # if k == 'returns':
    #             #     input_dict[k] = v.reshape(-1, 1, *v.shape[2:])[sample_idx].reshape(self.minibatch_size, *v.shape[2:])
    #             # else:
    #             input_dict[k] = v.reshape(-1, 2, *v.shape[1:])[sample_idx].reshape(self.minibatch_size*2, *v.shape[1:])
    #             #print("------", k, input_dict[k].shape)
                
    #     if (end >= self.batch_size//2):
    #         self._shuffle_idx_buf()

    #     return input_dict
    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]
        #print('$$$$$$$$$$$',start,end,self.minibatch_size)
        input_dict = {}
        disc_k = ['global_amp_obs','global_amp_masks','global_amp_obs_demo','global_amp_demo_masks']
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                if k in disc_k:
                    input_dict[k] = v
                else:
                    input_dict[k] = v[sample_idx]
                
        if (end >= self.batch_size):
            self._shuffle_idx_buf()

        return input_dict

    def _shuffle_idx_buf(self):
        self._idx_buf[:] = torch.randperm(self.batch_size//2)
        return
    

class DualDataset(AMPDataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):
        super().__init__(batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len)
        return
    
    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]

        #print("minibatch size:", self.minibatch_size)
        #print("sample_idx:", sample_idx)

        input_dict = {}
        disc_k = ['amp_obs','amp_obs_replay','amp_obs_demo_in','amp_obs_demo_out','amp_obs_demo_mask_in', \
                  'amp_obs_demo_mask_out','text_latents_trans']
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                #print('++++++', k, v.shape)
                if k in disc_k:
                    input_dict[k] = v
                else:
                    input_dict[k] = v[sample_idx]
                #input_dict[k] = v.reshape(-1, 2, *v.shape[1:])[sample_idx].reshape(self.minibatch_size*2, *v.shape[1:])
                #print("------", k, input_dict[k].shape)
                
        if (end >= self.batch_size//2):
            self._shuffle_idx_buf()

        return input_dict