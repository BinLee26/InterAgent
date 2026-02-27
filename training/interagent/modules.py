import logging
import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
logger = logging.getLogger(__name__)
from training.interagent.interdit_utils  import *

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps, pos_emb):
        time_embeddings = self.time_embed(pos_emb(timesteps))
        return time_embeddings


class TransformerForDiffusion(nn.Module):
    def __init__(self,
                 action_dim: int,
                 horizon: int,
                 T_obs: int = None,
                 T_action: int=None,
                 proprioception_dim: int = 0,
                 exteroception_dim: int = 0,
                 n_layer: int = 8,
                 n_head: int = 12,
                 n_emb: int = 768,
                 n_emb_ig_seq: int = 32,
                 p_drop_attn: float = 0.1,
                 training: bool = False
                 ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.n_emb = n_emb

        self.T_obs = T_obs
        self.T_action = T_action
        self.proprioception_dim = proprioception_dim

        self.exteroception_dim = exteroception_dim

        self.training = training
        # compute number of tokens for main trunk and condition encoder

        # input embedding stem
        self.input_action_emb = nn.Linear(self.action_dim, n_emb)
        self.input_proprioception_emb = nn.Linear(self.proprioception_dim,n_emb)
        self.input_exteroception_emb = nn.Linear(self.exteroception_dim,n_emb)
        # cond encoder

        self.embed_text = nn.Linear(768, n_emb)



        self.exteroception_emb_recent4 = nn.Sequential(
            nn.Linear(3, 36),
            nn.Mish(),
            nn.Linear(36, 36))

        # self.exteroception_emb_recent4 = nn.Sequential(
        #     nn.Linear(45, 100),
        #     nn.Mish(),
        #     nn.Linear(100, 100))


        self.proprioception_emb_recent4 = nn.Sequential(
            nn.Linear(self.proprioception_dim, 1024),
            nn.Mish(),
            nn.Linear(1024, n_emb)
        )

        self.exteroception_emb_faraway12 = nn.Sequential(
            nn.Linear(3, 36),
            nn.Mish(),
            nn.Linear(36, 36)
        ) # edge version

        # self.exteroception_emb_faraway12 = nn.Sequential(
        #     nn.Linear(45, 100),
        #     nn.Mish(),
        #     nn.Linear(100, 100)
        # )

        self.proprioception_emb_faraway12 = nn.Sequential(
            nn.Linear(self.proprioception_dim, 1024),
            nn.Mish(),
            nn.Linear(1024, n_emb)
        )

        self.cond_pos_emb_recent4 = nn.Parameter(torch.zeros(1, 4, n_emb))
        self.cond_pos_emb_faraway12 = nn.Parameter(torch.zeros(1, 12, n_emb))

        self.cond_pos_emb_recent4_ig_seq = nn.Parameter(torch.zeros(1, 4*15*15, 36)) #nn.Parameter(torch.zeros(1, 4*15*15, 36))#nn.Parameter(torch.zeros(1, 4*225, n_emb_ig_seq))
        self.cond_pos_emb_faraway12_ig_seq = nn.Parameter(torch.zeros(1, 12*15*15, 36)) #nn.Parameter(torch.zeros(1, 12*15*15, 36))#nn.Parameter(torch.zeros(1, 12*225, n_emb_ig_seq))

        self.sequence_pos_encoder = PositionalEncoding(n_emb, dropout=0)
        self.sequence_pos_encoder_time = PositionalEncoding(n_emb, dropout=0)
        self.embed_timestep = TimestepEmbedder(n_emb, self.sequence_pos_encoder_time)


        self.decoder = nn.ModuleList()
        for i in range(n_layer):
            self.decoder.append(TransformerBlock(nhead=n_head, latent_dim=self.n_emb, dropout=p_drop_attn, ff_size=4 * n_emb))
        self.out_action = zero_module(FinalLayer(self.n_emb, self.action_dim))
        self.out_proprioception = zero_module(FinalLayer(self.n_emb, self.proprioception_dim))
        self.out_exteroception = zero_module(FinalLayer(self.n_emb, self.exteroception_dim))

        # attention mask
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
        # therefore, the upper triangle should be -inf and others (including diag) should be 0.
        sz = self.T_action  #self.horizon - self.T_obs  # TAKARA :T
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask_row = torch.cat((mask, mask, mask), dim=0)
        mask_all = torch.cat((mask_row, mask_row, mask_row), dim=1)
        self.register_buffer("mask", mask_all)


        self.device = 'cuda'

        # init
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_optim_groups(self, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules) and not fpn.startswith("hist_obs_encoder") :
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules) and not fpn.startswith("hist_obs_encoder"):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

                elif fpn.startswith("hist_obs_encoder"):
                    no_decay.add(fpn)

        no_decay.add("cond_pos_emb_recent4")
        no_decay.add("cond_pos_emb_faraway12")
        no_decay.add("cond_pos_emb_recent4_ig_seq")
        no_decay.add("cond_pos_emb_faraway12_ig_seq")


        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def forward(self,
                sample: torch.Tensor,
                text: torch.Tensor,
                timestep,
                obs,
                **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        obs: (B,T',obs_dim)
        output: (B,T,input_dim)
        """

        ###############################################################################

        # process input

        text_emb = self.embed_text(text)
        time_emb = self.embed_timestep(timestep)


        proprioception_a_recent4 = obs[:,12:16,:self.proprioception_dim]
        proprioception_b_recent4= obs[:,12:16,self.proprioception_dim+self.exteroception_dim:self.proprioception_dim+self.exteroception_dim+self.proprioception_dim]

        exteroception_a_recent4 = obs[:, 12:16, self.proprioception_dim:self.proprioception_dim+self.exteroception_dim]
        exteroception_b_recent4 = obs[:, 12:16, self.proprioception_dim+self.exteroception_dim+self.proprioception_dim:]

        exteroception_a_recent4_ig_raw = exteroception_a_recent4.reshape(-1,4,225,3)#reshape(-1,4,15,45) #(-1,4,225,3)
        exteroception_b_recent4_ig_raw = exteroception_b_recent4.reshape(-1,4,225,3)#(-1,4,15,45) #(-1,4,225,3)

        exteroception_a_recent4_ig_seq = exteroception_a_recent4_ig_raw.reshape(-1,4*225,3)#(-1,4*15,45) #(-1,4*225,3)reshape(-1,4*15,45)
        exteroception_b_recent4_ig_seq = exteroception_b_recent4_ig_raw.reshape(-1,4*225,3)#(-1,4*15,45) #(-1,4*225,3)reshape(-1,4*15,45)


        proprioception_a_faraway12 = obs[:, 0:12,:self.proprioception_dim]
        proprioception_b_faraway12 = obs[:, 0:12,self.proprioception_dim+self.exteroception_dim:self.proprioception_dim+self.exteroception_dim+self.proprioception_dim]

        exteroception_a_faraway12 = obs[:, 0:12, self.proprioception_dim:self.proprioception_dim+self.exteroception_dim]
        exteroception_b_faraway12 = obs[:, 0:12, self.proprioception_dim+self.exteroception_dim+self.proprioception_dim:]

        exteroception_a_faraway12_ig_raw = exteroception_a_faraway12.reshape(-1,12,225,3)
        exteroception_b_faraway12_ig_raw = exteroception_b_faraway12.reshape(-1,12,225,3)

        exteroception_a_faraway12_ig_seq = exteroception_a_faraway12_ig_raw.reshape(-1,12*225,3)
        exteroception_b_faraway12_ig_seq = exteroception_b_faraway12_ig_raw.reshape(-1,12*225,3)

        proprioception_emb_a_recent4 = self.proprioception_emb_recent4(proprioception_a_recent4) + self.cond_pos_emb_recent4
        proprioception_emb_b_recent4 = self.proprioception_emb_recent4(proprioception_b_recent4) + self.cond_pos_emb_recent4

        exteroception_emb_a_recent4 = self.exteroception_emb_recent4(exteroception_a_recent4_ig_seq) + self.cond_pos_emb_recent4_ig_seq
        exteroception_emb_b_recent4 = self.exteroception_emb_recent4(exteroception_b_recent4_ig_seq) + self.cond_pos_emb_recent4_ig_seq


        proprioception_emb_a_faraway12 = self.proprioception_emb_faraway12(proprioception_a_faraway12) + self.cond_pos_emb_faraway12
        proprioception_emb_b_faraway12 = self.proprioception_emb_faraway12(proprioception_b_faraway12) + self.cond_pos_emb_faraway12

        exteroception_emb_a_faraway12 = self.exteroception_emb_faraway12(exteroception_a_faraway12_ig_seq) + self.cond_pos_emb_faraway12_ig_seq
        exteroception_emb_b_faraway12 = self.exteroception_emb_faraway12(exteroception_b_faraway12_ig_seq) + self.cond_pos_emb_faraway12_ig_seq

        cond_embeddings = time_emb + text_emb


        x_action_a, x_action_b = sample[...,self.proprioception_dim+self.exteroception_dim:self.proprioception_dim+self.exteroception_dim+self.action_dim], sample[...,2*self.proprioception_dim+2*self.exteroception_dim+self.action_dim:]
        x_proprioception_a, x_proprioception_b = sample[...,:self.proprioception_dim], sample[..., self.proprioception_dim+self.exteroception_dim+self.action_dim:2*self.proprioception_dim+self.exteroception_dim+self.action_dim]
        x_exteroception_a, x_exteroception_b = sample[...,self.proprioception_dim:self.proprioception_dim+self.exteroception_dim], sample[..., 2*self.proprioception_dim+self.exteroception_dim+self.action_dim:2*self.proprioception_dim+2*self.exteroception_dim+self.action_dim]

        action_a_emb = self.input_action_emb(x_action_a)
        action_b_emb = self.input_action_emb(x_action_b)
        h_a_prev = self.sequence_pos_encoder(action_a_emb)
        h_b_prev = self.sequence_pos_encoder(action_b_emb)

        proprioception_a_emb = self.input_proprioception_emb(x_proprioception_a)
        proprioception_b_emb = self.input_proprioception_emb(x_proprioception_b)
        proprioception_h_a_prev = self.sequence_pos_encoder(proprioception_a_emb)
        proprioception_h_b_prev = self.sequence_pos_encoder(proprioception_b_emb)

        exteroception_a_emb = self.input_exteroception_emb(x_exteroception_a)
        exteroception_b_emb = self.input_exteroception_emb(x_exteroception_b)
        exteroception_h_a_prev = self.sequence_pos_encoder(exteroception_a_emb)
        exteroception_h_b_prev = self.sequence_pos_encoder(exteroception_b_emb)

        for i, block in enumerate(self.decoder):
            proprioception_h_a,exteroception_h_a, h_a = block(h_a_prev, h_b_prev,proprioception_h_a_prev,proprioception_h_b_prev,exteroception_h_a_prev,exteroception_h_b_prev, cond_embeddings, proprioception_emb_a_recent4,exteroception_emb_a_recent4,  proprioception_emb_a_faraway12,exteroception_emb_a_faraway12, self.mask)
            proprioception_h_b,exteroception_h_b, h_b = block(h_b_prev, h_a_prev,proprioception_h_b_prev,proprioception_h_a_prev,exteroception_h_b_prev,exteroception_h_a_prev, cond_embeddings, proprioception_emb_b_recent4,exteroception_emb_b_recent4,  proprioception_emb_b_faraway12,exteroception_emb_b_faraway12, self.mask)
            proprioception_h_a_prev,exteroception_h_a_prev, h_a_prev = proprioception_h_a,exteroception_h_a, h_a
            proprioception_h_b_prev,exteroception_h_b_prev, h_b_prev = proprioception_h_b,exteroception_h_b, h_b

        output_action_a = self.out_action(h_a)
        output_action_b = self.out_action(h_b)

        output_proprioception_a = self.out_proprioception(proprioception_h_a)
        output_proprioception_b = self.out_proprioception(proprioception_h_b)

        output_exteroception_a = self.out_exteroception(exteroception_h_a)
        output_exteroception_b = self.out_exteroception(exteroception_h_b)

        output_a = torch.cat([output_proprioception_a,output_exteroception_a, output_action_a], dim=-1)
        output_b = torch.cat([output_proprioception_b,output_exteroception_b, output_action_b], dim=-1)

        output = torch.cat([output_a, output_b], dim=-1)

        return output

