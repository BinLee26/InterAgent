import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class AdaLN(nn.Module):

    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            # nn.Linear(embed_dim, latent_dim, bias=True),
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale[:, None]) + shift[:, None]
        return h


class Hybrid_Attention(nn.Module):
    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout,
                 embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.xf_latent_dim = xf_latent_dim
        self.dropout = nn.Dropout(dropout)
        self.num_branches = 3
        self.head_dim = latent_dim // self.num_head

        self.norm_list = nn.ModuleList([
            AdaLN(latent_dim, embed_dim) for _ in range(self.num_branches)
        ])
        self.xf_norm_list = nn.ModuleList([
            AdaLN(xf_latent_dim, embed_dim) for _ in range(self.num_branches)
        ])

        self.q_proj_list = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(self.num_branches)
        ])
        self.k_proj_list = nn.ModuleList([
            nn.Linear(xf_latent_dim, latent_dim) for _ in range(self.num_branches)
        ])
        self.v_proj_list = nn.ModuleList([
            nn.Linear(xf_latent_dim, latent_dim) for _ in range(self.num_branches)
        ])

        self.out_proj_list = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(self.num_branches)
        ])

    def forward(self, x_list,xf_list, emb=None,attn_mask=None, key_padding_mask=None):
        """
        x_list: [x1, x2, x3], each (B,T,D)
        xf_list: [xf1, xf2, xf3], each (B,N,D)
        """
        B = x_list[0].shape[0]
        T0,T1,T2 = x_list[0].shape[1], x_list[1].shape[1], x_list[2].shape[1]
        h = self.num_head

        Q_all, K_all, V_all = [], [], []

        # ---- branch-wise QKV ----
        for i in range(self.num_branches):
            x = self.norm_list[i](x_list[i], emb)
            xf = self.xf_norm_list[i](xf_list[i], emb)
            Q = self.q_proj_list[i](x).view(B, x_list[i].shape[1], h, self.head_dim).transpose(1, 2)  # [B,h,T,36]
            K = self.k_proj_list[i](xf).view(B, xf_list[i].shape[1], h, self.head_dim).transpose(1, 2)  # [B,h,N,36]
            V = self.v_proj_list[i](xf).view(B, xf_list[i].shape[1], h, self.head_dim).transpose(1, 2)  # [B,h,N,36]

            Q_all.append(Q)
            K_all.append(K)
            V_all.append(V)

        # ---- concat along sequence dim ----
        Q = torch.cat(Q_all, dim=2)
        K = torch.cat(K_all, dim=2)
        V = torch.cat(V_all, dim=2)

        # ---- scaled dot-product attention ----
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask

        if key_padding_mask is not None:
            # mask: [B, N_total] or [B, 1, 1, N_total]
            attn_logits = attn_logits.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, Q.shape[2], h * self.head_dim)

        out1 = self.out_proj_list[0](out[:,:T0,:])
        out2 = self.out_proj_list[1](out[:,T0:T0+T1,:])
        out3 = self.out_proj_list[2](out[:,T0+T1:T0+T1+T2,:])

        return out1, out2, out3

class Cross_Attention(nn.Module):
    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout,
                 embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.xf_latent_dim = xf_latent_dim
        self.dropout = nn.Dropout(dropout)
        self.num_branches = 3
        self.head_dim = latent_dim // self.num_head

        self.norm_list = nn.ModuleList([
            AdaLN(latent_dim, embed_dim) for _ in range(self.num_branches)
        ])
        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)

        self.q_proj_list = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(self.num_branches)
        ])

        self.k_proj = nn.Linear(xf_latent_dim, latent_dim)
        self.v_proj = nn.Linear(xf_latent_dim, latent_dim)

        self.out_proj_list = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(self.num_branches)
        ])

    def forward(self, x_list, xf, emb=None, key_padding_mask=None):
        """
        x_list: [x1, x2, x3], each (B,T,D)
        xf_list: [xf1, xf2, xf3], each (B,N,D)
        """
        B = x_list[0].shape[0]
        T0,T1,T2 = x_list[0].shape[1], x_list[1].shape[1], x_list[2].shape[1]
        h = self.num_head

        Q_all = []

        # ---- branch-wise QKV ----
        for i in range(self.num_branches):
            x = self.norm_list[i](x_list[i], emb)
            Q = self.q_proj_list[i](x).view(B, x_list[i].shape[1], h, self.head_dim).transpose(1, 2)  # [B,h,T,36]
            Q_all.append(Q)
        xf = self.xf_norm(xf, emb)
        K = self.k_proj(xf).view(B, xf.shape[1], h, self.head_dim).transpose(1, 2)  # [B,h,N,36]
        V = self.v_proj(xf).view(B, xf.shape[1], h, self.head_dim).transpose(1, 2)  # [B,h,N,36]

        # ---- concat along sequence dim ----
        Q = torch.cat(Q_all, dim=2)  # [B,h,T_total,dim]

        # ---- scaled dot-product attention ----
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) #[B, h, T,N]

        if key_padding_mask is not None:
            # mask: [B, N_total] or [B, 1, 1, N_total]
            attn_logits = attn_logits.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # [B,h,T_total,36]
        out = out.transpose(1, 2).contiguous().view(B, Q.shape[2], h * self.head_dim)

        out1 = self.out_proj_list[0](out[:,:T0,:])
        out2 = self.out_proj_list[1](out[:,T0:T0+T1,:])
        out3 = self.out_proj_list[2](out[:,T0+T1:T0+T1+T2,:])

        return out1, out2, out3

#edge version final
class SparseSeqKVAttention_V7(nn.Module):
    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout, topk,
                 embed_dim=None, gumbel_tau=1.0):
        super().__init__()
        self.num_head = num_head
        self.topk = topk
        self.latent_dim = latent_dim
        self.xf_latent_dim = xf_latent_dim
        self.dropout = nn.Dropout(dropout)
        self.gumbel_tau = gumbel_tau
        self.num_branches = 3  # x1,x2,x3

        # 各分支独立归一化
        self.norm_list = nn.ModuleList([
            AdaLN(latent_dim, embed_dim) for _ in range(self.num_branches)
        ])

        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)

        self.q_proj_list = nn.ModuleList([
            nn.Linear(latent_dim, num_head * 36) for _ in range(self.num_branches)
        ])

        self.k_proj = nn.ModuleList([
            nn.Linear(xf_latent_dim, 36) for _ in range(num_head)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(xf_latent_dim, 36) for _ in range(num_head)
        ])

        self.out_proj_list = nn.ModuleList([
            nn.Linear(num_head * 36, latent_dim) for _ in range(self.num_branches)
        ])

    def forward(self, x_list, xf, emb=None, training=True):
        """
        x_list: [x1, x2, x3], each (B,T,D)
        xf: (B,N,D)
        """
        B = x_list[0].shape[0]
        T0,T1,T2 = x_list[0].shape[1], x_list[1].shape[1], x_list[2].shape[1]
        h = self.num_head
        N = xf.shape[1]
        assert N % h == 0, f"序列长度 N={N} 必须能整除 num_head={h}"
        nh = N // h

        Q_all = []

        # ---- branch-wise projection ----
        for i in range(self.num_branches):
            x = x_list[i]
            x_norm = self.norm_list[i](x, emb)
            Q = self.q_proj_list[i](x_norm).view(B, x_list[i].shape[1], h, 36).transpose(1, 2)  # (B,h,T,36)
            Q_all.append(Q)

        Q = torch.cat(Q_all, dim=2)  # (B,h,T_total,36)

        xf_norm = self.xf_norm(xf, emb)
        xf_split = xf_norm.view(B, h, nh, self.xf_latent_dim)

        k_weight = torch.stack([proj.weight for proj in self.k_proj], dim=0)
        k_bias = torch.stack([proj.bias for proj in self.k_proj], dim=0)
        K = torch.einsum('bhnd,hkd->bhnk', xf_split, k_weight) + k_bias[:, None, :]

        v_weight = torch.stack([proj.weight for proj in self.v_proj], dim=0)
        v_bias = torch.stack([proj.bias for proj in self.v_proj], dim=0)
        V = torch.einsum('bhnd,hkd->bhnk', xf_split, v_weight) + v_bias[:, None, :]

        # ---- attention ----
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(36)

        # ---- top-k sparse ----
        self.topk = min(self.topk, K.shape[-2])
        if training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(attn_logits) + 1e-9) + 1e-9)
            logits = attn_logits + gumbel_noise
        else:
            logits = attn_logits

        soft_weights = F.softmax(logits, dim=-1)
        topk_idx = torch.topk(logits, self.topk, dim=-1).indices
        topk_mask = torch.zeros_like(logits)
        topk_mask.scatter_(-1, topk_idx, 1.0)
        mask_logits = logits.masked_fill(topk_mask == 0, float('-inf'))
        hard_weights = F.softmax(mask_logits, dim=-1)
        attn = hard_weights + soft_weights - soft_weights.detach()
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Q.shape[2], h * 36)

        out1 = self.out_proj_list[0](out[:,:T0,:])
        out2 = self.out_proj_list[1](out[:,T0:T0+T1,:])
        out3 = self.out_proj_list[2](out[:,T0+T1:T0+T1+T2,:])

        return out1, out2, out3

class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim,out_dim, dropout, embed_dim=None):
        super().__init__()
        self.norm = AdaLN(latent_dim, embed_dim)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = zero_module(nn.Linear(ffn_dim, out_dim, bias=True))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        if emb is not None:
            x_norm = self.norm(x, emb)
        else:
            x_norm = x
        y = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        return y


class FinalLayer(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.linear = zero_module(nn.Linear(latent_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.linear(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 seq_len=4,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl
        self.seq_len = seq_len

        self.sa_block = Hybrid_Attention(latent_dim,latent_dim, num_heads, dropout,latent_dim)

        self.ca_block = Hybrid_Attention(latent_dim, latent_dim, num_heads, dropout,latent_dim)

        self.proprioception_block_recent = Cross_Attention(latent_dim, latent_dim, num_heads, dropout,latent_dim)
        self.exteroception_block_recent = SparseSeqKVAttention_V7(latent_dim, 36,30, dropout,15,latent_dim)# 1/2

        self.proprioception_block_faraway = Cross_Attention(latent_dim, latent_dim, num_heads, dropout,latent_dim)
        self.exteroception_block_faraway = SparseSeqKVAttention_V7(latent_dim, 36, 30, dropout,45,latent_dim)#  1/2

        self.action_ffn = FFN(latent_dim, ff_size, latent_dim, dropout,latent_dim)
        self.proprioception_ffn = FFN(latent_dim, ff_size, latent_dim, dropout,latent_dim)
        self.exteroception_ffn = FFN(latent_dim, ff_size, latent_dim, dropout,latent_dim)

# 0815version
    def forward(self, x_action, y_action,x_proprioception,y_proprioception,x_exteroception, y_exteroception, emb=None, proprioception_emb_recent=None, exteroception_emb_recent=None, proprioception_emb_faraway=None, exteroception_emb_faraway=None,attn_mask=None):
        p0, e0, a0 = self.sa_block([x_proprioception,x_exteroception, x_action],[x_proprioception,x_exteroception, x_action], emb, attn_mask)
        p0 = p0 + x_proprioception
        e0 = e0 + x_exteroception
        a0 = a0 + x_action
        p1, e1, a1 = self.proprioception_block_recent([p0,e0,a0], proprioception_emb_recent, emb)
        p1 = p1 + p0
        e1 = e1 + e0
        a1 = a1 + a0
        p2, e2, a2 = self.proprioception_block_faraway([p1,e1,a1], proprioception_emb_faraway, emb)
        p2 = p2 + p1
        e2 = e2 + e1
        a2 = a2 + a1
        p3,e3,a3 = self.exteroception_block_recent([p2,e2,a2], exteroception_emb_recent, emb)
        p3 = p3 + p2
        e3 = e3 + e2
        a3 = a3 + a2
        p4,e4,a4 = self.exteroception_block_faraway([p3,e3,a3], exteroception_emb_faraway, emb)
        p4 = p4 + p3
        e4 = e4 + e3
        a4 = a4 + a3
        p5,e5,a5 = self.ca_block([x_proprioception,x_exteroception, x_action], [y_proprioception,y_exteroception, y_action], emb, attn_mask)
        p5 = p5 + p4
        e5 = e5 + e4
        a5 = a5 + a4
        out_proprioception = self.proprioception_ffn(p5, emb)
        out_proprioception = p5 + out_proprioception
        out_exteroception = self.exteroception_ffn(e5, emb)
        out_exteroception = e5 + out_exteroception
        out_action = self.action_ffn(a5, emb)
        out_action = a5 + out_action

        return out_proprioception, out_exteroception, out_action

