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

import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim
from isaacgym.torch_utils import *

import learning.replay_buffer as replay_buffer
import learning.common_agent as common_agent 
import learning.ase_agent as ase_agent
import learning.ase_models as ase_models
import learning.ase_network_builder as ase_network_builder

from tensorboardX import SummaryWriter

class DMAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):           
        super().__init__(base_name, config)

        self._task_size = self.vec_env.env.task.get_task_obs_size()
        self._num_motions = self.vec_env.env.task.get_num_motions()
        self._num_envs = self.vec_env.env.task.get_num_envs()

        self.is_moe = config['MoE']
        
        return
    
    def init_tensors(self):
        super().init_tensors()
        self.tensor_list += ['rewards']
        return
    
    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs = self.obs['obs']

        obs, rewards, dones, infos = self.vec_env.step(actions)
        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos


    def play_steps(self):
        self.set_eval()
        
        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                #print("updating", k, res_dict[k].shape)
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float()
            terminated = terminated.reshape(-1, 1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1

            all_done_indices = self.dones[::self.num_agents].nonzero(as_tuple=False)
            done_indices = all_done_indices
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values
        
        self.debug_reward(mb_rewards)
        
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        
        return batch_dict  

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)


        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'is_moe' : self.is_moe
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)
            
            a_loss = torch.mean(a_loss)
            c_loss = torch.mean(c_loss)
            b_loss = torch.mean(b_loss)
            entropy = torch.mean(entropy)
            
            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss

            if self.is_moe:
                aux_loss = res_dict['aux_loss']
                loss += aux_loss
                # print('a:',a_loss,'c:',c_loss,'b:',b_loss,'aux:',aux_loss)
                
            a_clip_frac = torch.mean(a_info['actor_clipped'].float())
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        if self.is_moe:
            self.train_result.update({'aux_loss':aux_loss})

        return
    
    def env_reset(self, env_ids=None):
        obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs
    
    def _build_net_config(self):
        config = super()._build_net_config()
        #print('===',config['input_shape'][0])
        return config

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs = obs_dict['obs']
        processed_obs = self._preproc_obs(obs)
        value = self.model.a2c_network.eval_critic(processed_obs)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value
    
    def preprocess_actions(self, actions):
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        if not self.is_tensor_obses:
            clamped_actions = clamped_actions.cpu().numpy()
        return clamped_actions

    def _setup_action_space(self):
        super()._setup_action_space()
        # todo introduce device instead of cuda()
        self.actions_low = torch.cat([self.actions_low, self.actions_low], dim=-1).to(self.ppo_device)
        self.actions_high = torch.cat([self.actions_high, self.actions_high], dim=-1).to(self.ppo_device)
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['rewards'] = batch_dict['rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        ig_reward_std, ig_reward_mean = torch.std_mean(train_info['rewards'])
        self.writer.add_scalar('info/ig_reward_mean', ig_reward_mean.item(), frame)
        self.writer.add_scalar('info/ig_reward_std', ig_reward_std.item(), frame)
        if self.is_moe:
            self.writer.add_scalar('losses/aux_loss', torch_ext.mean_list(train_info['aux_loss']).item(), frame)
        return
    
    def debug_reward(self, rewards):
        #print('reward shape:', rewards.shape)
        print('total/env0 rewards:', rewards.mean())