import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, TwoWayAttentionLayer, FullAttentionPatch
from layers.Embed import DataEmbedding_inverted
import numpy as np
from utils.hippo import HiPPO_LegS
import torch.fft


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model

        self.kernel_size = configs.kernel_size
        self.n_heads = configs.n_heads

        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    TwoWayAttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, self.kernel_size, self.n_heads, mode=configs.struct_mode),
                    configs.d_model,
                    self.kernel_size,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    mode=configs.struct_mode
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projector = nn.Linear(self.d_model, configs.pred_len, bias=True)
        self.patch_proj = nn.Linear(configs.seq_len, configs.d_model)

        self.embed = nn.Linear(configs.d_hippo + configs.d_model, configs.d_model)
        self.W_channels = nn.Parameter(nn.init.uniform_(torch.empty(1, configs.enc_in, configs.d_model), -0.02, 0.02),
                                       requires_grad=True).to(configs.device)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, hippo):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # x_enc = self.gen_inputs(x_enc, hippo)
        x_enc = x_enc.permute(0, 2, 1)  # [B, N, L]
        x_enc = self.patch_proj(x_enc)
        x_enc = self.embed(torch.cat([x_enc, hippo], dim=-1))  # [B, N, D]
        x_enc = x_enc + self.W_channels

        enc_out, attns = self.encoder(x_enc, attn_mask=None)  # [B, N, D]
        # enc_out = enc_out[:, :, -1, :] # [B, N, D]

        if not self.training:
            ts = enc_out[0, -1, :]
            ts = ts.cpu().numpy()
            np.save('ts_time.npy', ts)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)  # filter the covariates [B, L, N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns  # [B, L, N]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, hippo, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, hippo)
        dec_out = dec_out[:, -self.pred_len:, :]
        return dec_out, attns  # [B, L, D]

    def gen_inputs(self, x, hippo):
        # x [B, L, N]  hippo [B, N, D]
        x = x.permute(0, 2, 1)  # [B, N, L]
        mix_feature = self.patch_proj(x)
        mix_feature = self.embed(torch.cat([mix_feature, hippo], dim=-1))  # [B, N, D]
        mix_feature = mix_feature + self.W_channels

        return mix_feature
