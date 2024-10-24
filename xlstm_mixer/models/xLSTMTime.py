from typing import Literal
from torch import nn
from xlstm_mixer.lit.enums import Task

# taken from https://github.dev/muslehal/xLSTMTime
from einops import rearrange
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

import argparse

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat, pack, unpack
from ..layers.Autoformer_EncDec import series_decomp
from ..layers.StandardNorm import Normalize as RevIN
from .base_model import BaseModel

mlstm_config = mLSTMBlockConfig()
# slstm_config = sLSTMBlockConfig()


class xLSTMTime(BaseModel):

    def __init__(
        self, pred_len: int, seq_len: int, enc_in: int, n2: int = 256
    ) -> None:
        super().__init__(pred_len=pred_len, seq_len=seq_len, enc_in=enc_in)

        
        # config_s = xLSTMBlockStackConfig(
        #     mlstm_block=mlstm_config,
        #     slstm_block=slstm_config,
        #     num_blocks=3,
        #     embedding_dim=256,
        #     add_post_blocks_norm=True,
        #     # _block_map=1,
        #     slstm_at="all",
        #     context_length=862 if enc_in == 862 else 336,
        # )
        config_m = xLSTMBlockStackConfig(
            mlstm_block=mlstm_config,
            # slstm_block=slstm_config,
            num_blocks=1,
            embedding_dim=256,
            add_post_blocks_norm=True,
            _block_map=1,
            # slstm_at="all",
            context_length=862,
        )

        self.enc_in = enc_in
        self.context_points = seq_len
        self.target_points = pred_len
        self.n2 = n2
        self.embedding_dim = n2

        # self.weighting_after = WeightingLayer(self.target_points)
        # self.weighting_before = WeightingLayer(self.embedding_dim)
        self.batch_norm = nn.BatchNorm1d(self.enc_in)

        # Decomposition Kernel Size
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.context_points, self.target_points)
        self.Linear_Trend = nn.Linear(self.context_points, self.target_points)

        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.context_points)
            * torch.ones([self.target_points, self.context_points])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.context_points)
            * torch.ones([self.target_points, self.context_points])
        )

        self.mm = nn.Linear(self.target_points, self.n2)
        self.mm2 = nn.Linear(self.embedding_dim, self.target_points)
        self.mm3 = nn.Linear(self.context_points, self.n2)

        reversible_instance_norm_affine = False
        use_reversible_instance_norm = True
        self.reversible_instance_norm = (
            RevIN(enc_in, affine=reversible_instance_norm_affine)
            if use_reversible_instance_norm
            else None
        )


        # if xlstm_kind == "s":
        #     self.xlstm_stack = xLSTMBlockStack(config_s)
        # else:
        self.xlstm_stack = xLSTMBlockStack(config_m)

    def forecast(self, x_enc, x_mark_enc):
        # print(x.shape)
        # batch time variate
        x = self.reversible_instance_norm(x_enc, "norm")

        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        seasonal_output = self.Linear_Seasonal(seasonal_init) 
        trend_output = self.Linear_Trend(trend_init) 

        x = seasonal_output + trend_output

        x = self.mm(x) 
        x = self.batch_norm(x)
        x = self.xlstm_stack(x)  

        x = self.mm2(x)  

        x = rearrange(x, "b v n  -> b n v")
        x = self.reversible_instance_norm(x, "denorm")

        return x

    
