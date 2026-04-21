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
import os
import yaml

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from utils.motion_lib_il import MotionLibIl
from utils.motion_lib_il_ig import MotionLibIlIg
from learning.dm_network_builder import DMBuilder

from env.tasks.humanoid_il_dm_v2 import HumanoidIlDMv2
from env.tasks.humanoid_il_dm_v3 import HumanoidIlDMv3
from env.tasks.humanoid_il_ig_v2 import HumanoidIlIgv2
from env.tasks.humanoid_il_ig_v3 import HumanoidIlIgv3
from env.tasks.humanoid_amp_task_dual import HumanoidAMPTaskDual
from env.tasks.humanoid_il_getup import HumanoidIlGetup
from utils import torch_utils

import numpy as np

class HumanoidIlMCP(HumanoidIlGetup):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.num_prim = cfg["env"].get("num_prim", 3)
        self.discrete_mcp = cfg["env"].get("discrete_mcp", False)
        self.models_path = cfg["env"].get("models_path", [])
        
        # Initialize parent class first
        super().__init__(cfg=cfg,
                        sim_params=sim_params,
                        physics_engine=physics_engine,
                        device_type=device_type,
                        device_id=device_id,
                        headless=headless)
        
        # Load pre-trained policies
        with open(os.path.join(os.getcwd(), cfg['env']['prim_config']), 'r') as f:
            prim_config = yaml.load(f, Loader=yaml.SafeLoader)
            prim_config_params = prim_config['params']

        self.running_mean_std = []
        self.actors = []
        for model_path in self.models_path:
            actor, running_mean_std = self._load_policy(prim_config_params, model_path)
            self.actors.append(actor)
            self.running_mean_std.append(running_mean_std)
            
        # Initialize MCP controller
        # self.mcp = MCP(
        #     num_prim=self.num_prim,
        #     actors=self.actors,
        #     obs_dim=self.get_obs_size(),
        #     discrete_mcp=self.discrete_mcp,
        #     device=self.device
        # )
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        self._num_actions = self.num_prim
        return

    def _load_policy(self, config_params, model_path):
        """Load a pre-trained policy network"""
        network_params = config_params['network']
        network_builder = DMBuilder()
        network_builder.load(network_params)
        # Prepare kwargs for network construction
        prim_net_config = {
            'actions_num': 28,
            'input_shape': (self.get_obs_size(),),
            'num_seqs': self.num_agents,
            'value_size': 1,
        }
        
        # Build the network
        actor = network_builder.build('dual',**prim_net_config)
        checkpoint = torch_ext.load_checkpoint(model_path)
        
        # Fix state dict keys by removing 'a2c_network.' prefix
        state_dict = checkpoint['model']
        print('checkpoint keys: ', checkpoint['running_mean_std'].keys())
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('a2c_network.', '')
            new_state_dict[new_key] = value
            
        actor.load_state_dict(new_state_dict)

        running_mean_std = RunningMeanStd(self.get_obs_size()).to(self.device)
        running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        running_mean_std.eval()
        
        actor.to(self.device)
        actor.eval()
        print("Loaded policy from {:s}".format(model_path))
        return actor, running_mean_std
        
    def step(self, weights):
        """
        Step the environment with MCP-generated actions
        
        Args:
            weights: Weights for combining primitive policies
        """
        with torch.no_grad():
  # Pack into dict for DMNetwork
            
            # Use MCP to get actions
            if self.discrete_mcp:
                max_idx = torch.argmax(weights, dim=1)
                weights = torch.nn.functional.one_hot(max_idx, num_classes=self.num_prim).float()
            
            # Get actions from each primitive policy and combine them
            primitive_actions = []
            for actor, running_mean_std in zip(self.actors, self.running_mean_std):
                cur_obs = self.obs_buf.reshape(self.num_envs*self.num_agents, -1)
                cur_obs = torch.clamp(cur_obs, min=-5.0, max=5.0)
                proc_obs = running_mean_std(cur_obs)
                mu, _, _, _ = actor({'obs': proc_obs})  # DMNetwork returns (mu, sigma, value, states)
                primitive_actions.append(mu)
            
            x_all = torch.stack(primitive_actions, dim=1)
            #print('x_all: ', x_all.shape, weights.shape)
            actions = torch.sum(weights[:, :, None] * x_all, dim=1)
            
        # Apply actions to the environment
        self.pre_physics_step(actions)
        
        # Step physics and render
        self._physics_step()
        
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            
        # Compute observations, rewards, resets, etc.
        self.post_physics_step()

        return
        
    def get_mcp_details(self):
        mcp_details = {}
        mcp_details['num_prim'] = self.num_prim
        mcp_details['discrete_mcp'] = self.discrete_mcp
        return mcp_details 