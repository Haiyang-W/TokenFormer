# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import LayerNorm as LayerNorm
from .fused_layer_norm import MixedFusedLayerNorm
import torch.nn.functional as F


def get_norm(neox_args):
    if neox_args.norm == "rmsnorm":
        norm = RMSNorm
        eps = neox_args.rms_norm_epsilon
    elif neox_args.norm == "layernorm":
        eps = neox_args.layernorm_epsilon
        norm = MixedFusedLayerNorm if neox_args.layernorm_fusion else LayerNorm
    elif neox_args.norm == "scalenorm":
        eps = neox_args.scalenorm_epsilon
        norm = ScaleNorm
    elif neox_args.norm == "layernorm_nonparam":
        eps = neox_args.layernorm_epsilon
        norm = LayerNorm_NonParam
    else:
        raise ValueError(f"norm {neox_args.norm} not recognized")
    return norm, eps

def get_final_norm(neox_args):
    if neox_args.final_norm == "rmsnorm":
        norm = RMSNorm
        eps = neox_args.rms_norm_epsilon
    elif neox_args.final_norm == "layernorm":
        eps = neox_args.layernorm_epsilon
        norm = MixedFusedLayerNorm if neox_args.layernorm_fusion else LayerNorm
    elif neox_args.final_norm == "scalenorm":
        eps = neox_args.scalenorm_epsilon
        norm = ScaleNorm
    elif neox_args.final_norm == "layernorm_nonparam":
        eps = neox_args.layernorm_epsilon
        norm = LayerNorm_NonParam
    else:
        raise ValueError(f"norm {neox_args.norm} not recognized")
    return norm, eps


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias

        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(dim))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        dtype = x.dtype
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return (self.scale * x_normed).to(dtype)


class ScaleNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g

class LayerNorm_NonParam(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.num_channels = dim
        self.eps = eps

    def forward(self, x):
        x = F.layer_norm(x, normalized_shape=(self.num_channels,), eps=self.eps)
        return x
