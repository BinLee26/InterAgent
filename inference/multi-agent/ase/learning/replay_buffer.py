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
import random

class ReplayBuffer():
    def __init__(self, buffer_size, device):
        self._head = 0
        self._total_count = 0
        self._buffer_size = buffer_size
        self._device = device
        self._data_buf = None
        self._sample_idx = torch.randperm(buffer_size)
        self._sample_head = 0

        return

    def reset(self):
        self._head = 0
        self._total_count = 0
        self._reset_sample_idx()
        return

    def get_buffer_size(self):
        return self._buffer_size

    def get_total_count(self):
        return self._total_count

    def store(self, data_dict):
        if (self._data_buf is None):
            self._init_data_buf(data_dict)

        n = next(iter(data_dict.values())).shape[0]
        #print('++++buf',data_dict.values(), n)
        buffer_size = self.get_buffer_size()
        #print('*****store',n,buffer_size)
        assert(n <= buffer_size)

        for key, curr_buf in self._data_buf.items():
            #print('*****k',key)
            curr_n = data_dict[key].shape[0]
            assert(n == curr_n)

            store_n = min(curr_n, buffer_size - self._head)
            curr_buf[self._head:(self._head + store_n)] = data_dict[key][:store_n]    
        
            remainder = n - store_n
            if (remainder > 0):
                curr_buf[0:remainder] = data_dict[key][store_n:]  

        self._head = (self._head + n) % buffer_size
        self._total_count += n

        return

    def sample(self, n):
        total_count = self.get_total_count()
        buffer_size = self.get_buffer_size()

        idx = torch.arange(self._sample_head, self._sample_head + n)
        idx = idx % buffer_size
        rand_idx = self._sample_idx[idx]
        if (total_count < buffer_size):
            rand_idx = rand_idx % self._head

        samples = dict()
        for k, v in self._data_buf.items():
            samples[k] = v[rand_idx]

        self._sample_head += n
        if (self._sample_head >= buffer_size):
            self._reset_sample_idx()

        return samples

    def _reset_sample_idx(self):
        buffer_size = self.get_buffer_size()
        self._sample_idx[:] = torch.randperm(buffer_size)
        self._sample_head = 0
        return

    def _init_data_buf(self, data_dict):
        buffer_size = self.get_buffer_size()
        self._data_buf = dict()

        for k, v in data_dict.items():
            v_shape = v.shape[1:]
            #print('*******v_shape',v.shape)
            self._data_buf[k] = torch.zeros((buffer_size,) + v_shape, device=self._device)
            #print('*******v_shape',self._data_buf[k].shape)
        return
    
class Replay_Buffer_id:
    def __init__(self, buffer_size, device):
        self._buffer_size = buffer_size
        self._device = device
        self._id_buffers = {}

    def reset(self):
        self._id_buffers = {}

    def get_buffer_size(self):
        return self._buffer_size

    def store(self, data_dict, item_id):
        if item_id not in self._id_buffers:
            self._init_buffer(data_dict, item_id)

        buffer = self._id_buffers[item_id]
        buffer.store(data_dict)

    # def sample_by_id(self, item_id, sample_size, sample_from_others=False):
    #     if item_id not in self._id_buffers:
    #         return None

    #     buffer = self._id_buffers[item_id]
    #     samples = buffer.sample(sample_size)

    #     if sample_from_others:
    #         other_samples = self._sample_from_other_buffers(item_id, sample_size)
    #         for k in samples.keys():
    #             samples[k] = torch.cat([samples[k], other_samples[k]], dim=0)

    #     return samples

    def _init_buffer(self, data_dict, item_id):
        buffer_size = self.get_buffer_size()
        buffer = ReplayBuffer(buffer_size, self._device)
        self._id_buffers[item_id] = buffer
        return buffer
    
    def sample_by_id_list(self, id_list, sample_size, size_per_sample):
        sample_in = {}
        sample_out = {}
        #print('++++idlist',id_list)
        id_list = id_list.tolist()
        num_ids = len(id_list)
        #print('+++buf',self._id_buffers[0].sample(sample_size).keys())
        for k in self._id_buffers[0].sample(sample_size).keys():
            sample_in[k] = torch.zeros((sample_size, size_per_sample), dtype=torch.float32, device=self._device)
            sample_out[k] = torch.zeros((sample_size, size_per_sample), dtype=torch.float32, device=self._device)

        for i, item_id in enumerate(id_list):
            #print('++++enumerate',i,item_id)
            if item_id not in self._id_buffers.keys():
                continue

            buffer = self._id_buffers[item_id]
            samples = buffer.sample(1)

            for k in samples.keys():
                sample_in[k][i] = samples[k].clone()

            other_id = self._sample_other_id(item_id)
            #print('++++outid',item_id, other_id)
            other_buffer = self._id_buffers[other_id]
            other_samples = other_buffer.sample(1)
            for k in other_samples.keys():
                sample_out[k][i] = other_samples[k].clone()

        return sample_in, sample_out
    
    def _sample_other_id(self, item_id):
        other_id = [id for id in self._id_buffers.keys() if id != item_id]
        if not other_id:
            return None
        return random.choice(other_id)
