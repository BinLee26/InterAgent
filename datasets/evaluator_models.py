import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

import random
import torch.distributed as dist

from models import *


loss_ce = nn.CrossEntropyLoss()
class InterCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.motion_encoder = MotionEncoder(cfg)

        self.latent_dim = self.latent_dim

        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype
        self.latent_scale = nn.Parameter(torch.Tensor([1]))

        set_requires_grad(self.token_embedding, False)

        self.textTransEncoder = clip_model.transformer
        self.text_ln = clip_model.ln_final #nn.LayerNorm(768)

        set_requires_grad(self.textTransEncoder, False)
        set_requires_grad(self.text_ln, False)

        self.out = nn.Linear(768, 512)

        self.clip_training = "text_"
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')

    def compute_loss(self, batch):
        losses = {}
        losses["total"] = 0

        # compute clip losses
        batch = self.encode_text(batch)
        batch = self.encode_motion(batch)

        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        losses.update(clip_losses)
        losses["total"] += mixed_clip_loss

        return losses["total"], losses

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        if 1:
            for d in self.clip_training.split('_')[:1]:
                if d == 'image':
                    features = self.clip_model.encode_image(batch['images']).float()  # preprocess is done in dataloader
                elif d == 'text':
                    features = batch['text_emb']
                motion_features = batch['motion_emb']
                # normalized features
                features_norm = features / features.norm(dim=-1, keepdim=True)
                motion_features_norm = motion_features / motion_features.norm(dim=-1, keepdim=True)

                logit_scale = self.latent_scale ** 2
                logits_per_motion = logit_scale * motion_features_norm @ features_norm.t()
                logits_per_d = logits_per_motion.t()

                batch_size = motion_features.shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=motion_features.device)

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                mixed_clip_loss += clip_mixed_loss

        return mixed_clip_loss, clip_losses

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def encode_motion(self, batch):
        batch["mask"] = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"]).to(batch["motions"].device)
        batch.update(self.motion_encoder(batch))
        batch["motion_emb"] = batch["motion_emb"] / batch["motion_emb"].norm(dim=-1, keepdim=True) * self.latent_scale

        return batch

    def encode_text(self, batch):
        device = next(self.parameters()).device
        raw_text = batch["text"]

        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.textTransEncoder(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.text_ln(x).type(self.dtype)

        # out = self.textTransEncoder(pe_tokens)
        # out = self.text_ln(out)

        out = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        out = self.out(out)

        batch['text_emb'] = out
        batch["text_emb"] = batch["text_emb"] / batch["text_emb"].norm(dim=-1, keepdim=True) * self.latent_scale

        return batch

class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)

def all_gather(tensor):
    return AllGatherFunction.apply(tensor)

class MultiInterCLIP(InterCLIP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        for d in self.clip_training.split('_')[:1]:
            if d == 'image':
                features = self.clip_model.encode_image(batch['images']).float()
            elif d == 'text':
                features = batch['text_emb']
            motion_features = batch['motion_emb']

            # normalize
            features_norm = features / features.norm(dim=-1, keepdim=True)
            motion_features_norm = motion_features / motion_features.norm(dim=-1, keepdim=True)

            # gather across GPUs
            all_features = all_gather(features_norm)
            all_motion_features = all_gather(motion_features_norm)

            # DDP info
            batch_size = motion_features.shape[0]
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # construct global ground truth index
            ground_truth = torch.arange(rank * batch_size, (rank + 1) * batch_size, device=features.device)

            # logits
            logit_scale = self.latent_scale ** 2
            logits_per_motion = logit_scale * motion_features_norm @ all_features.t()
            logits_per_d = logit_scale * features_norm @ all_motion_features.t()

            ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
            ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
            clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

            clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
            clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
            clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
            mixed_clip_loss += clip_mixed_loss

        return mixed_clip_loss, clip_losses

