'''
File implementing the higher-level policy API (e.g. querying actions and computing losses).
Abstracts away the lower-level architecture details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from pdp.utils.normalizer import LinearNormalizer


class DiffusionPolicy(nn.Module):
    def __init__(
            self,
            model,
            noise_scheduler: DDPMScheduler,
            **kwargs
    ):
        super().__init__()

        self.model = model
        self.obs_dim = self.model.proprioception_dim
        self.action_dim = self.model.action_dim
        self.T_obs = self.model.T_obs
        self.T_action = self.model.T_action #self.model.horizon - self.model.T_obs

        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.normalizer = None # set by set_normalizer

    @property
    def T_range(self):
        return self.T_obs + self.T_action - 1

    def get_optim_groups(self, weight_decay):
        return self.model.get_optim_groups(weight_decay)

    # ========= inference  ============

    def classifier_free_sample(self,text_feature, cond_data, cond_mask, cond=None, guidance_scale=3.5):
        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[cond_mask] = cond_data[cond_mask]
            cond_output = model(trajectory,text_feature, t, cond)
            uncond_output = model(trajectory,torch.zeros_like(text_feature), t, cond)

            model_output = uncond_output + guidance_scale * (cond_output - uncond_output)

            # compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        trajectory[cond_mask] = cond_data[cond_mask]
        return trajectory

    def predict_action(self, obs_dict, text_feature):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert obs_dict['obs'].shape[1:] == (self.T_obs, self.obs_dim)
        nobs = self.normalizer.normalize(obs_dict)['obs']
        B, _, obs_dim = nobs.shape

        # Handle different ways of passing observation
        cond = nobs[:, :self.T_obs]
        shape = (B, self.T_action, self.action_dim)
        cond_data = torch.zeros(size=shape, device=nobs.device, dtype=nobs.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run sampling
        nsample = self.classifier_free_sample(text_feature,cond_data, cond_mask, cond=cond)

        # Unnormalize prediction and extract action

        nresult = {'action': nsample}
        result = self.normalizer.unnormalize(nresult)
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def get_optimizer(self, weight_decay, learning_rate, betas):
        return self.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas)
        )

    def forward(self, epoch, batch):
        return self.compute_loss(epoch,batch)

    def mask_condition(self, cond, cond_mask_prob):
        bs = cond.shape[0]
        keep_mask = torch.bernoulli((1.0 - cond_mask_prob) * torch.ones(bs, device=cond.device))
        keep_mask = keep_mask.view([bs] + [1] * (cond.ndim - 1))
        mask_cond = cond * keep_mask
        return mask_cond

    def compute_loss(self, epoch,batch):
        nbatch = self.normalizer.normalize({
            'obs': batch['obs'],
            'action': batch['action']
        })
        obs = nbatch['obs']
        action = nbatch['action']
        ##############--------------------------------------------------------

        ################------------------------------------------------------
        text_feature = batch['text_feature']

        text_feature = self.mask_condition(text_feature, 0.1)

        obs_raw = obs[:, :self.T_obs]
        cond = obs_raw
        start = self.T_obs - 1
        end = start + self.T_action

        #---------------------
        trajectory = torch.cat([obs[:,self.T_obs:,:898], action[:,start:end,:28], obs[:,self.T_obs:,898:], action[:,start:end,28:]], dim=2)

        # generate impainting mask
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        B = trajectory.shape[0]
        K = self.noise_scheduler.config.num_train_timesteps

        # Sample a random timestep for each image
        timesteps = torch.randint(0, K, (B,), device=trajectory.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, text_feature, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')  # (B, 4, 952)
        loss = loss * loss_mask.type(loss.dtype)


        proprioception_loss_part1 = loss[:, :, 0:223]
        proprioception_loss_part2 = loss[:, :, 926:1149]
        proprioception_loss = torch.cat([proprioception_loss_part1, proprioception_loss_part2], dim=-1)  # (B, 4, 896)

        exteroception_loss_part1 = loss[:, :, 223:898]
        exteroception_loss_part2 = loss[:, :, 1149:1824]
        exteroception_loss = torch.cat([exteroception_loss_part1, exteroception_loss_part2], dim=-1)  # (B, 4, 896)

        act_loss_part1 = loss[:, :, 898:926]
        act_loss_part2 = loss[:, :, 1824:1852]
        action_loss = torch.cat([act_loss_part1, act_loss_part2], dim=-1)

        proprioception_loss = proprioception_loss.mean()
        exteroception_loss = exteroception_loss.mean()
        action_loss = action_loss.mean()

        proprioception_weight = 1.0
        exteroception_weight = 1.0
        action_weight = 1.0
        total_loss = proprioception_weight * proprioception_loss + exteroception_weight * exteroception_loss + action_weight * action_loss

        return total_loss, proprioception_loss,exteroception_loss, action_loss

