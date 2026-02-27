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
import sys
from datetime import datetime
from gym import spaces
import numpy as np
import os
import torch 
import yaml
from collections import defaultdict
import joblib
from collections import deque
import pickle

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import learning.common_player as common_player
# import learning.ase_models as ase_models
# import learning.ase_network_builder as ase_network_builder
# import learning.ase_players as ase_players
#
# import learning.amp_models as amp_models
# import learning.amp_network_builder as amp_network_builder
# import learning.amp_players as amp_players

from learning.rollout_utils import load_checkpoint
import dill
import collections
import clip
clip_model, _ = clip.load("ViT-L/14@336px", 'cuda',jit=False)
clip_model.eval() # frozen clip model

# from mld.data.get_data import get_datasets
# from mld.models.get_model import get_model
# from omegaconf import OmegaConf

class DMPlayer(common_player.CommonPlayer):
    def __init__(self, config):
        
        super().__init__(config)
        
        self._task_size = self.env.task.get_task_obs_size()
        self._obs_size = self.env.task.get_obs_size()
        self._action_size = self.env.task.get_action_size()
        self.eval = self.player_config['eval']
        self.save_succ_rate = self.player_config['save_succ_rate']
        self.eval_step = 0
        self.games_num = 200000
        self._recovery_steps = self.player_config['player_recovery_step']
        
        self.tracking_data_save_path = self.player_config['tracking_data_save_path']
        
        if self.eval:
            self.num_motions = self.env.task.get_num_motions()
            self._num_envs = self.env.task.get_num_envs()
            self.get_motion_info_list()
            print("++motions++:", self._motion_names[:5])
            print("++num frames++:",list(self._motion_num_frames.items())[:5])
            self.motion_success_rates = defaultdict(lambda: {"terminate_count": 0, "total_reset": 0})
            self.cur_collect_motion_id = 0
            self.cur_collect_motion_step = 0

            # self.max_steps = 10000000 // 4
            self.eval_save_path = self.player_config['eval_save_path']
        
            if self.env.task.collect_dataset:
                self.file_count = 0
                self.max_num_trajs = 10
                self.max_traj_steps = 1000
                self.data_obs_dim = self._obs_size - self._task_size
                self.env_buffers = [deque(maxlen=self.max_traj_steps) for _ in range(self._num_envs)]

                self.motion_data = {
                name: {
                    "obs": np.zeros((self.max_num_trajs, self._motion_num_frames[name]+self._recovery_steps-1, 2, self.data_obs_dim)),  # (50, 100, 2, obs_dim)
                    "clean_action": np.zeros((self.max_num_trajs, self._motion_num_frames[name]+self._recovery_steps-1, 2, self._action_size)),
                    "traj_count": 0,
                    "is_terminate": np.zeros((self.max_num_trajs,self._motion_num_frames[name]+self._recovery_steps-1)),
                    "is_reset": np.zeros((self.max_num_trajs, self._motion_num_frames[name]+self._recovery_steps-1)),
                    "rigid_body_pos": np.zeros((self.max_num_trajs, self._motion_num_frames[name]+self._recovery_steps-1, 2, 15, 3)),
                    "rigid_body_ros": np.zeros((self.max_num_trajs, self._motion_num_frames[name]+self._recovery_steps-1, 2, 15, 4)),
                    "text": ' '
                }
                for name in self._motion_names
                }

            self.payload = torch.load(open(
                "./checkpoints/checkpoint.ckpt",
                'rb'), pickle_module=dill)
            self.policy = load_checkpoint(self.payload)

            self.motion_file_path = self.env.task.cfg['args'].motion_file
            self.text_file_name = self.motion_file_path.split("/")[-1].split('_')[0] + '.txt'
            self.chosen_index = int(self.motion_file_path.split("/")[-1].split('_')[1].split('.')[0])
            self.text_fold = "./data"
            self.text_path = os.path.join(self.text_fold, self.text_file_name)
            with open(self.text_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
                self.text = self.lines[self.chosen_index - 1]

            with torch.no_grad():
                self.text_tokens = clip.tokenize(self.text, truncate=True).to('cuda')
                self.text_feature = clip_model.encode_text(self.text_tokens)
                self.text_feature = self.text_feature.to(torch.float32)


        return   

    def restore(self, fn):
        if (fn != 'Base'):
            super().restore(fn)
        return
    
    def get_motion_info_list(self):
        self._motion_names = []
        self._motion_num_frames = dict()
        for motion_id in range(self.num_motions):
            cur_motion = self.env.task.get_motion_name(motion_id)
            cur_motion_name = os.path.basename(cur_motion)
            cur_motion_name = cur_motion_name.split("_")[0]
            self._motion_names.append(cur_motion_name)
            self._motion_num_frames[cur_motion_name] = self.env.task.get_motion_num_frames(motion_id)

    def get_action(self, obs_dict, is_determenistic = False):
        obs = obs_dict['obs']

        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        proc_obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : proc_obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        
        current_action = current_action.detach()
        clamped_actions = torch.clamp(current_action, -1.0, 1.0)
        #clamped_actions = current_action
        # print(res_dict['values'])
        return clamped_actions

    def record_step(self, env_id, motion_id, obs, rbp, rbr, clean_action, env_action, is_reset, is_terminate):

        buffer = self.env_buffers[env_id]
        motion_name = self._motion_names[motion_id]
        max_traj_steps = self._motion_num_frames[motion_name] + self._recovery_steps - 2 #self._motion_num_frames[motion_name] + self._recovery_steps - 1

        
        buffer.append((obs,rbp, rbr, clean_action,int(is_reset.cpu().numpy()), int(is_terminate.cpu().numpy())))
        print("\n")
        print("motion name:", motion_name)
        print("checkrecover",len(buffer),max_traj_steps)
        print("is_terminate:", is_terminate)
        print("text:",self.text)
        print("chosen index:",self.chosen_index)
        if len(buffer) == max_traj_steps:
            #assert(is_reset)        
            record_idx = self.motion_data[motion_name]["traj_count"]
            if record_idx < self.max_num_trajs:
                #print("debug+++",motion_name,self.motion_data[motion_name]["obs"].shape, max_traj_steps)
                #print(len(buffer), reset)
                for i, (o, rbp_i,rbr_i,clean_a, is_re,is_termi) in enumerate(buffer):
                    self.motion_data[motion_name]["obs"][record_idx, i] = o
                    self.motion_data[motion_name]["clean_action"][record_idx, i] = clean_a
                    self.motion_data[motion_name]["is_reset"][record_idx, i] = is_re
                    self.motion_data[motion_name]["is_terminate"][record_idx, i] = is_termi
                    self.motion_data[motion_name]["rigid_body_pos"][record_idx, i] = rbp_i
                    self.motion_data[motion_name]["rigid_body_ros"][record_idx, i] = rbr_i
                self.motion_data[motion_name]["traj_count"] += 1
                self.motion_data[motion_name]["text"] = self.text

        #     buffer.clear()
        # elif is_reset:
        #     buffer.clear()

        return
            
        
    def _post_step(self, info):
        #print('+++++cur_motion:',self.env.task.get_motion_name(info['curr_motion_ids'][0]))
        if self.eval:
            self.eval_step += 1
            humanoid_env = self.env.task
            reset_buf = info['reset'][:,0]
            terminate_buf = info['terminate'][:,0]
            #print('+++++cur_motion:',humanoid_env.get_motion_name(info['curr_motion_ids'][0]))
            for motion_id, reset_flag, terminate_flag in zip(info['curr_motion_ids'], reset_buf, terminate_buf):
                motion_id = motion_id.item()
                cur_motion_name = self._motion_names[motion_id]
                #print(cur_motion_name)
                if reset_flag == 1:
                    self.motion_success_rates[cur_motion_name]["total_reset"] += 1
                    if terminate_flag == 1:
                        self.motion_success_rates[cur_motion_name]["terminate_count"] += 1

            for motion_id, stats in self.motion_success_rates.items():
                if stats["total_reset"] > 0:
                    stats["success_rate"] = 1 - (stats["terminate_count"] / stats["total_reset"])
                else:
                    stats["success_rate"] = None
                    
            if humanoid_env.collect_dataset:
                self.cur_collect_motion_step += 1
                ########## collect dataset #########
                for env_id in range(self._num_envs):
                    self.record_step(env_id, info['curr_motion_ids'][env_id], info['obs_buf'][env_id,:,:-self._task_size], 
                                    info['rigid_body_pos_t'][env_id], info['rigid_body_rot_t'][env_id], info['clean_actions'][env_id], info['actions'][env_id], reset_buf[env_id], terminate_buf[env_id] )
                
                ########## update to next motion ##############
                cur_collect_motion_name = self._motion_names[self.cur_collect_motion_id]
                if self.motion_data[cur_collect_motion_name]["traj_count"] == self.max_num_trajs:
                    self.save_collected_data(cur_collect_motion_name)
                    self.cur_collect_motion_id += 1
                    humanoid_env._motion_lib.set_cur_sample_idx(self.cur_collect_motion_id)
                    self.cur_collect_motion_step = 0
                
                ######## print progress ##############
                collected = 0
                total_expected = self.num_motions * self.max_num_trajs
                for data in self.motion_data.values():
                    collected += data["traj_count"]
                print(f'collecting progress:{collected}/{total_expected}, step {self.cur_collect_motion_step}')
                
                ######## finish ###########
                if all(data["traj_count"] >= self.max_num_trajs for data in self.motion_data.values()):
                    sys.exit()
                    return
                
            if self.save_succ_rate and self.eval_step >= 500:
                all_terminates = np.array([stats["terminate_count"] for stats in self.motion_success_rates.values()])
                all_resets = np.array([stats["total_reset"] for stats in self.motion_success_rates.values()])
                #print('++++allreset',all_terminates.sum(),all_resets.sum())
                total_success_rate = 1 - (all_terminates.sum() / all_resets.sum()) if all_resets.sum() > 0 else None

                print(f"Overall Success Rate: {total_success_rate:.3f}")
                self.eval_step = 0
                print('evaluating..')
                save_path = self.eval_save_path
                motion_names = self.motion_success_rates.keys()

                joblib.dump({"motion_success_rates": dict(self.motion_success_rates), "total_success_rate": total_success_rate, "num_frames": self._motion_num_frames}, save_path)
                print(f"Saved success rates to {save_path}")
        return

    def save_collected_data(self, motion_name):
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.tracking_data_save_path
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{motion_name}_{self.chosen_index}.pkl")
        
        with open(filename, "wb") as f:
            pickle.dump(self.motion_data[motion_name], f)
        print(f'+++++saved motion data to {filename}')
        return
    
    # def save_collected_data(self):
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     save_dir = "ase/tracking_dataset"
    #     os.makedirs(save_dir, exist_ok=True)
    #     filename = os.path.join(save_dir, f"motion_data_{timestamp}.pkl")
        
    #     with open(filename, "wb") as f:
    #         pickle.dump(self.motion_data, f)
    #     print(f'+++++saved motion data to {filename}')
    #     return

        
        
    # def run(self):
    #     n_games = self.games_num
    #     render = self.render_env
    #     n_game_life = self.n_game_life
    #     is_determenistic = self.is_determenistic
    #     sum_rewards = 0
    #     sum_steps = 0
    #     sum_game_res = 0
    #     n_games = n_games * n_game_life
    #     games_played = 0
    #     has_masks = False
    #     has_masks_func = getattr(self.env, "has_action_mask", None) is not None

    #     op_agent = getattr(self.env, "create_agent", None)
    #     if op_agent:
    #         agent_inited = True

    #     if has_masks_func:
    #         has_masks = self.env.has_action_mask()

    #     need_init_rnn = self.is_rnn
    #     for _ in range(n_games):
    #         if games_played >= n_games:
    #             break

    #         obs_dict = self.env_reset()
    #         batch_size = 1
    #         if len(obs_dict['obs'].size()) > len(self.obs_shape):
    #             batch_size = obs_dict['obs'].size()[0]
    #         self.batch_size = batch_size

    #         if need_init_rnn:
    #             self.init_rnn()
    #             need_init_rnn = False

    #         cr = torch.zeros(batch_size, dtype=torch.float32)
    #         steps = torch.zeros(batch_size, dtype=torch.float32)

    #         print_game_res = False

    #         done_indices = []

    #         for n in range(self.max_steps):
    #             obs_dict = self.env_reset(done_indices)

    #             if has_masks:
    #                 masks = self.env.get_action_mask()
    #                 action = self.get_masked_action(obs_dict, masks, is_determenistic)
    #             else:
    #                 action = self.get_action(obs_dict, is_determenistic)
    #             obs_dict, r, done, info = self.env_step(self.env, action)
    #             cr += r
    #             steps += 1
  
    #             self._post_step(info)

    #             if render:
    #                 self.env.render(mode = 'human')
    #                 time.sleep(self.render_sleep)

    #             all_done_indices = done[::self.num_agents].nonzero(as_tuple=False)
    #             done_indices = all_done_indices
    #             done_count = len(done_indices)
    #             games_played += done_count

    #             if done_count > 0:
    #                 if self.is_rnn:
    #                     for s in self.states:
    #                         s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

    #                 cur_rewards = cr[done_indices].sum().item()
    #                 cur_steps = steps[done_indices].sum().item()

    #                 cr = cr * (1.0 - done.float())
    #                 steps = steps * (1.0 - done.float())
    #                 sum_rewards += cur_rewards
    #                 sum_steps += cur_steps

    #                 game_res = 0.0
    #                 if isinstance(info, dict):
    #                     if 'battle_won' in info:
    #                         print_game_res = True
    #                         game_res = info.get('battle_won', 0.5)
    #                     if 'scores' in info:
    #                         print_game_res = True
    #                         game_res = info.get('scores', 0.5)
    #                 if self.print_stats:
    #                     if print_game_res:
    #                         print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
    #                     else:
    #                         print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

    #                 sum_game_res += game_res
    #                 if batch_size//self.num_agents == 1 or games_played >= n_games:
    #                     break
        
    #             done_indices = done_indices[:, 0]

    #     print(sum_rewards)
    #     if print_game_res:
    #         print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
    #     else:
    #         print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

    #     return
    
    # def env_step(self, env, obs_dict, action):
    #     if not self.is_tensor_obses:
    #         actions = actions.cpu().numpy()
            
    #     obs = obs_dict['obs']
    #     rewards = 0.0
    #     done_count = 0.0
    #     disc_rewards = 0.0
    #     for t in range(self._llc_steps):

    #         llc_actions = self._compute_llc_action(obs, action)
            
    #         obs, curr_rewards, curr_dones, infos = env.step(llc_actions)

    #         rewards += curr_rewards
    #         done_count += curr_dones

    #         amp_obs = infos['amp_obs']
    #         curr_disc_reward = self._calc_llc_disc_reward(amp_obs)
    #         curr_disc_reward = curr_disc_reward[0, 0].cpu().numpy()
    #         disc_rewards += curr_disc_reward

    #     rewards /= self._llc_steps
    #     dones = torch.zeros_like(done_count)
    #     dones[done_count > 0] = 1.0

    #     disc_rewards /= self._llc_steps
    #     #print("disc_reward", disc_rewards)

    #     if isinstance(obs, dict):
    #         obs = obs['obs']
    #     if obs.dtype == np.float64:
    #         obs = np.float32(obs)
    #     if self.value_size > 1:
    #         rewards = rewards[0]
    #     if self.is_tensor_obses:
    #         return obs, rewards.cpu(), dones.cpu(), infos
    #     else:
    #         if np.isscalar(dones):
    #             rewards = np.expand_dims(np.asarray(rewards), 0)
    #             dones = np.expand_dims(np.asarray(dones), 0)
    #         return torch.from_numpy(obs).to(self.device), torch.from_numpy(rewards), torch.from_numpy(dones), infos
    
    # def _build_llc(self, config_params, checkpoint_file):
    #     network_params = config_params['network']
    #     network_builder = ase_network_builder.ASEBuilder()
    #     network_builder.load(network_params)

    #     network = ase_models.ModelASEContinuous(network_builder)
    #     llc_agent_config = self._build_llc_agent_config(config_params, network)

    #     self._llc_agent = ase_players.ASEPlayer(llc_agent_config)
    #     self._llc_agent.restore(checkpoint_file)
    #     print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
    #     return

    # def _build_llc_agent_config(self, config_params, network):
    #     llc_env_info = copy.deepcopy(self.env_info)
    #     obs_space = llc_env_info['observation_space']
    #     #print('+++++llc_amp_space', obs_space)
    #     obs_size = obs_space.shape[0] - 225
    #     obs_size -= self._task_size
    #     llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])
    #     #print('+++++obs+++++', llc_env_info['observation_space'])
    #     llc_num_amp_obs = 1250 
    #     llc_env_info['amp_observation_space'] = spaces.Box(np.ones(llc_num_amp_obs) * -np.Inf, np.ones(llc_num_amp_obs) * np.Inf).shape
    #     #print('+++++amp+++++', llc_env_info['amp_observation_space'])
    #     llc_env_info['num_envs'] = self.env.task.num_envs
    #     config = config_params['config']
    #     config['network'] = network
    #     #config['num_actors'] = self.num_actors
    #     #config['features'] = {'observer' : self.algo_observer}
    #     config['env_info'] = llc_env_info
    #     #print('###########',self.num_actors)
    #     return config
    
    # def _setup_action_space(self):
    #     super()._setup_action_space()
    #     self.actions_num = self._latent_dim
    #     # todo introduce device instead of cuda()
    #     # self.actions_low = torch.cat([self.actions_low, self.actions_low], dim=-1).to(self.device)
    #     # self.actions_high = torch.cat([self.actions_high, self.actions_high], dim=-1).to(self.device)

    #     # self.actions_low = torch.cat([self.actions_low, self.actions_low], dim=-1).to(self.device)
    #     # self.actions_high = torch.cat([self.actions_high, self.actions_high], dim=-1).to(self.device)
    #     return
    

    # def init_tensors(self):
    #     super().init_tensors()

    #     del self.experience_buffer.tensor_dict['actions']
    #     del self.experience_buffer.tensor_dict['mus']
    #     del self.experience_buffer.tensor_dict['sigmas']

    #     batch_shape = self.experience_buffer.obs_base_shape
    #     self.experience_buffer.tensor_dict['actions'] = torch.zeros(batch_shape + (self._latent_dim,),
    #                                                             dtype=torch.float32, device=self.ppo_device)
    #     self.experience_buffer.tensor_dict['mus'] = torch.zeros(batch_shape + (self._latent_dim,),
    #                                                             dtype=torch.float32, device=self.ppo_device)
    #     self.experience_buffer.tensor_dict['sigmas'] = torch.zeros(batch_shape + (self._latent_dim,),
    #                                                             dtype=torch.float32, device=self.ppo_device)
        

        
    #     self.experience_buffer.tensor_dict['disc_rewards'] = torch.zeros((batch_shape[0],batch_shape[1]//2, 1),
    #                                                             dtype=torch.float32, device=self.ppo_device)
    #     self.tensor_list += ['disc_rewards']

    #     return
    

    # def _compute_llc_action(self, obs, actions):
    #     actions1 = actions[0::2]
    #     actions2 = actions[1::2]


    #     #print("action size check:", actions.shape, actions1.shape)
    #     llc_obs = self._extract_llc_obs(obs, 0)
    #     processed_obs = self._llc_agent._preproc_obs(llc_obs)

    #     z = torch.nn.functional.normalize(actions1, dim=-1)
    #     mu1, _ = self._llc_agent.model.a2c_network.eval_actor(obs=processed_obs, ase_latents=z)

    #     llc_obs = self._extract_llc_obs(obs, 1)
    #     processed_obs = self._llc_agent._preproc_obs(llc_obs)

    #     z = torch.nn.functional.normalize(actions2, dim=-1)
    #     mu2, _ = self._llc_agent.model.a2c_network.eval_actor(obs=processed_obs, ase_latents=z)
        

    #     llc_action = torch.cat([mu1.unsqueeze(1), mu2.unsqueeze(1)], dim=1)
    #     # llc_action = players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(llc_action, -1.0, 1.0))

    #     return llc_action
    

    # def _extract_llc_obs(self, obs, id):
    #     obs_size = obs.shape[-1]
    #     if id == 0:
    #         llc_obs = obs[0::2, :obs_size - self._task_size - 225]
    #     else:
    #         llc_obs = obs[1::2, :obs_size - self._task_size - 225]

    #     return llc_obs
    
    # def _calc_llc_disc_reward(self, amp_obs):
    #     disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
    #     return disc_reward