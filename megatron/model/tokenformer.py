# Copyright (c) 2024 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pkg_resources import packaging
from importlib.metadata import version

from .norms import get_norm
from megatron import mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists, get_fusion_type
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb_torch,
    apply_rotary_pos_emb,
    AliBi,
)
from megatron.model.fused_rope import (
    FusedRoPEFunc,
    fused_apply_rotary_pos_emb_cached,
)
from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention
from deepspeed.moe.layer import MoE

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

class Pattention(nn.Module):
    """Pattention Layer.
    """

    def __init__(
        self,
        neox_args,
        input_channels,
        output_channels,
        param_token_num,
        param_key_init_method,
        param_value_init_method,
    ):
        super().__init__()

        self.param_token_num = param_token_num
        self.param_key_dim = input_channels
        self.param_value_dim = output_channels
        self.norm_activation_type = neox_args.norm_activation_type
        
        self.key_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_key_dim)))
        self.value_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_value_dim)))
        
        param_key_init_method(self.key_param_tokens)
        param_value_init_method(self.value_param_tokens)
    
    def nonlinear_norm_func(self, inputs, normalize_type, dim=-1):
        if normalize_type == 'softmax': 
            # NOTE: softmax = exp_l1_norm
            # outputs = F.softmax(inputs, dim=dim) * inputs.shape[dim]
            nonlinear_outputs = torch.exp(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True) * inputs.shape[dim]
            outputs = norm_outputs
        elif normalize_type == 'gelu_l2_norm':
            nonlinear_outputs = F.gelu(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True) * math.sqrt(nonlinear_outputs.shape[dim])
            outputs = norm_outputs
        elif normalize_type == 'l2_norm_gelu':
            norm_outputs = inputs / torch.norm(inputs, p=2, dim=dim, keepdim=True) * math.sqrt(inputs.shape[dim])
            nonlinear_outputs = F.gelu(norm_outputs)
            outputs = nonlinear_outputs
        else:
            raise NotImplementedError
        return outputs

    def forward(self, inputs, dropout_p=0.0, router_index=None, attn_mask=None, scale=None):

        query = inputs
        if router_index is None:
            # not MoE mode
            key, value = self.key_param_tokens, self.value_param_tokens
        else:
            key, value = self.key_param_tokens[router_index], self.value_param_tokens[router_index]
        
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 if scale is None else scale 
        # just for gelu nonlinear, set torch.zeros for softmax
        attn_bias = torch.ones(L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # just for gelu nonlinear, set -inf for softmax
                attn_bias.masked_fill_(attn_mask.logical_not(), 0)
            else:
                raise NotImplementedError

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        # just for gelu nonlinear, set attn_weight += attn_bias for softmax
        attn_weight *= attn_bias
        # modified softmax
        attn_weight = self.nonlinear_norm_func(attn_weight, self.norm_activation_type, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value

        return output

class ParallelLinear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        neox_args,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
        is_last_layer=False,
    ):
        super().__init__()
        parallelism = neox_args.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # rescale params only called if neox_args.use_mup = True, despite it not being included here
            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class ParallelSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        param_key_init_method,
        param_value_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):
        super().__init__()
        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )
        self.pos_emb = neox_args.pos_emb

        self.sliding_window_width = neox_args.sliding_window_width

        self.num_kv_heads_per_partition = self.num_attention_heads_per_partition
        self.kv_hidden_size = neox_args.hidden_size

        self.query = Pattention(
            neox_args=neox_args,
            input_channels=neox_args.hidden_size,
            output_channels=neox_args.hidden_size,
            param_token_num=neox_args.qkv_slot_num,
            param_key_init_method=param_key_init_method,
            param_value_init_method=param_value_init_method,
        )
        self.key = Pattention(
            neox_args=neox_args,
            input_channels=neox_args.hidden_size,
            output_channels=neox_args.hidden_size,
            param_token_num=neox_args.qkv_slot_num,
            param_key_init_method=param_key_init_method,
            param_value_init_method=param_value_init_method,
        )
        self.value = Pattention(
            neox_args=neox_args,
            input_channels=neox_args.hidden_size,
            output_channels=neox_args.hidden_size,
            param_token_num=neox_args.qkv_slot_num,
            param_key_init_method=param_key_init_method,
            param_value_init_method=param_value_init_method,
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        if neox_args.use_mup:
            self.norm_factor = self.hidden_size_per_attention_head

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                neox_args.num_attention_heads,
                neox_args.model_parallel_size,
                mpu.get_model_parallel_rank(),
            )

        # TODO: this arg shouldn't need to be passed in - get from neox_args
        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_emb = RotaryEmbedding(
                dim,
                base=neox_args.rotary_emb_base,
                max_seq_len=neox_args.seq_length,
                precision=neox_args.params_dtype,
                save_inv_freqs=neox_args.rotary_save_freqs_buffer,
            )
        else:
            self.rotary_emb = None

        self.rope_fusion = neox_args.rope_fusion
        self.attention_type = "flash"
        self.use_flash_attention = self.attention_type == "flash"
        self.use_triton = (
            self.use_flash_attention
            and self.pos_emb == "alibi"
            and (
                not packaging.version.Version(version("flash-attn"))
                >= packaging.version.Version("2.4.0.post1")
            )
        )

        from flash_attn.flash_attn_interface import (
            flash_attn_func,
            flash_attn_varlen_func,
        )
        from flash_attn.flash_attn_triton import (
            flash_attn_func as flash_attn_unpadded_unpacked_func_triton,
        )

        self.flash_triton_fn = flash_attn_unpadded_unpacked_func_triton
        self.flash_qkv_fn = flash_attn_func
        self.flash_varlen_qkv_fn = flash_attn_varlen_func
        self.dropout_p = neox_args.attention_dropout
        self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output.
        self.proj = Pattention(
                neox_args=neox_args,
                input_channels=neox_args.hidden_size,
                output_channels=neox_args.hidden_size,
                param_token_num=neox_args.proj_slot_num,
                param_key_init_method=param_key_init_method,
                param_value_init_method=param_value_init_method,
            )

    def flash_attention(self, query_layer, key_layer, value_layer):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        if self.use_flash_attention and not self.use_triton:

            # [sk, b, np, hn] -> [b, sk, np, hn] -> [b * sk, 1, np, hn]
            key_layer = key_layer.transpose(0, 1).reshape(
                output_size[0], output_size[3], self.num_kv_heads_per_partition, -1
            )
            value_layer = value_layer.transpose(0, 1).reshape(
                output_size[0], output_size[3], self.num_kv_heads_per_partition, -1
            )

            # [sq, b, np, hn] -> [b, sq, np, hn]
            query_layer = query_layer.transpose(0, 1).reshape(
                output_size[0], output_size[2], output_size[1], -1
            )

            # only pass in window_size or alibi_slopes kwarg
            # if we use Sliding Window Attention / AliBi.
            # Flash attn defaults to (-1,-1), or
            # does not have this kwarg prior to v2.3.0
            extra_kwargs = (
                {"window_size": (self.sliding_window_width, -1)}
                if self.sliding_window_width is not None
                else {}
            )
            if self.pos_emb == "alibi":
                extra_kwargs["alibi_slopes"] = self.alibi_embed.slopes.to(
                    query_layer.device
                ).to(torch.float32)

            if not self.training:
                batch_size = output_size[0]
                max_seqlen_q = output_size[2]
                max_seqlen_k = output_size[3]

                cu_seqlens_q = torch.arange(
                    0,
                    (batch_size + 1) * max_seqlen_q,
                    step=max_seqlen_q,
                    dtype=torch.int32,
                    device=query_layer.device,
                )

                cu_seqlens_k = torch.arange(
                    0,
                    (batch_size + 1) * max_seqlen_k,
                    step=max_seqlen_k,
                    dtype=torch.int32,
                    device=key_layer.device,
                )

                q_shape = query_layer.shape
                k_shape = key_layer.shape
                v_shape = value_layer.shape
                is_causal = max_seqlen_q == max_seqlen_k
                output = self.flash_varlen_qkv_fn(
                    query_layer.reshape(
                        (q_shape[0] * q_shape[1], q_shape[2], q_shape[3])
                    ),
                    key_layer.reshape(
                        (k_shape[0] * k_shape[1], k_shape[2], k_shape[3])
                    ),
                    value_layer.reshape(
                        (v_shape[0] * v_shape[1], v_shape[2], v_shape[3])
                    ),
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=None,
                    causal=is_causal,
                    **extra_kwargs,
                )
                output = output.reshape(q_shape)
            else:
                output = self.flash_qkv_fn(
                    query_layer,
                    key_layer,
                    value_layer,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=None,
                    causal=True,
                    **extra_kwargs,
                )

            matmul_result = output
            # [b, sq, np, hn] -> [b, np, sq, hn]
            matmul_result = matmul_result.transpose(1, 2)

        else:
            # we still use Triton if using AliBi with flash-attn<2.4.0.post1.

            # [sq, b, np, hn] -> [b, sq, np, hn]
            sq = query_layer.size(0)
            b = query_layer.size(1)
            sk = key_layer.size(0)

            query_layer = query_layer.transpose(0, 1)
            key_layer = key_layer.transpose(0, 1)
            value_layer = value_layer.transpose(0, 1)

            bias = self.alibi_embed.bias(sq, sk, query_layer.device, query_layer.dtype)
            bias = bias.unsqueeze(0).tile((b, 1, 1, 1))

            matmul_result = self.flash_triton_fn(
                query_layer, key_layer, value_layer, bias=bias, causal=True
            )
            matmul_result = matmul_result.transpose(1, 2)

        return matmul_result

    def forward(self, hidden_states, attention_mask, layer_past=None):

        # hidden_states: [sq, b, h]
        sq, b, h = hidden_states.size()

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> 3 [sq, b, np, hn]
        query_layer = self.query(hidden_states).view(sq, b, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        key_layer = self.key(hidden_states).view(sq, b, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        value_layer = self.value(hidden_states).view(sq, b, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims :],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims :],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer

            seq_len = key_layer.shape[0]
            offset = 0
            if exists(layer_past) and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len) # [223, 1, 1, 16], [223, 1, 1, 16],

            if self.rope_fusion:
                query_layer, key_layer = (
                    fused_apply_rotary_pos_emb_cached(rot, cos, sin)
                    for rot in [query_rot, key_rot]
                )
            else:
                if self.bf16:
                    apply_rotary_fn = apply_rotary_pos_emb_torch
                else:
                    apply_rotary_fn = apply_rotary_pos_emb
                query_layer, key_layer = apply_rotary_fn(
                    query_rot, key_rot, cos, sin, offset=offset
                )

            if exists(self.rotary_ndims):
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        context_layer = self.flash_attention(query_layer, key_layer, value_layer)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        output = self.proj(context_layer)

        if self.use_cache:
            output = [output, present]

        return output

class ParallelTokenformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):

        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)

        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.use_cache = use_cache

        self.hidden_dropout = neox_args.hidden_dropout
        self.bias_dropout_fusion = neox_args.bias_dropout_fusion
        self.mlp_type = neox_args.mlp_type

        # Self attention.
        self.attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            param_key_init_method=init_method,
            param_value_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            use_cache=self.use_cache,
            rotary=rotary,
        )

        self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)

        args = neox_args
        # Feed forward neural network
        self.mlp = Pattention(
            neox_args=neox_args,
            input_channels=neox_args.hidden_size,
            output_channels=neox_args.hidden_size,
            param_token_num=neox_args.ffn_slot_num,
            param_key_init_method=init_method,
            param_value_init_method=output_layer_init_method,
        )

        self.layer_past = None  # used to cache k/v pairs in inference

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = (
                bias_dropout_add_fused_train
                if self.training
                else bias_dropout_add_fused_inference
            )
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def forward(self, x, attention_mask, layer_past=None):
        layer_past = layer_past if layer_past is not None else self.layer_past
        bias_dropout_fn = self._get_bias_dropout()

        residual = x

        attention_output = self.attention(
            self.input_layernorm(x), attention_mask, layer_past=layer_past
        )
        if self.use_cache:
            attention_output, presents = attention_output
            self.layer_past = presents
        with torch.enable_grad():
            # Otherwise just apply dropout + residual
            attention_output = (
                torch.nn.functional.dropout(
                    attention_output,
                    p=self.hidden_dropout,
                    training=self.training,
                )
                + residual
            )

        layernorm_output = self.post_attention_layernorm(attention_output)
        mlp_bias = torch.tensor(
            0.0, device=layernorm_output.device, dtype=layernorm_output.dtype
        )

        mlp_output = self.mlp(layernorm_output)

        with torch.enable_grad():
            output = bias_dropout_fn(
                mlp_output,
                bias=mlp_bias.expand_as(attention_output),
                residual=attention_output,
                prob=self.hidden_dropout,
            )

        return output

class ParallelTokenformerLayerPipe(ParallelTokenformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        output = super().forward(hidden_states, attention_mask)
        # auxiliary output
        self.last_moe_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        return output, attention_mask


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "ParallelLinearPipe expects a single argument - hidden_states"
        hidden_state = args
        logits, bias = super().forward(hidden_state)
        return logits


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        assert not isinstance(
            args, tuple
        ), "NormPipe should only receive a single tensor as input"
        return self.norm(args)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)
