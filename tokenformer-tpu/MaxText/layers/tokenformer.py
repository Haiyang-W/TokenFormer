"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module


from typing import Any, Optional, Tuple, Sequence

import jax
from jax.sharding import Mesh
from jax import lax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from flax import linen as nn
from flax.typing import PRNGKey
from flax.linen.module import merge_param


from layers import attentions
from layers import initializers
from layers import linears
from layers import models
from layers import quantizations

AttentionOp = attentions.AttentionOp

import common_types

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV

DenseGeneral = linears.DenseGeneral
NdInitializer = initializers.NdInitializer
Initializer = initializers.Initializer
nd_dense_init = initializers.nd_dense_init
Quant = quantizations.AqtQuantization
KVQuant = quantizations.KVQuant


class DropPath(nn.Module):
  """Jax implementation of Stochastic Depth
  """
  rate: float
  broadcast_dims: Sequence[int] = ()
  deterministic: Optional[bool] = None
  rng_collection: str = 'dropout'

  @nn.compact
  def __call__(
    self,
    inputs,
    deterministic: Optional[bool] = None,
    rng: Optional[PRNGKey] = None,
  ):
    """Applies a random dropout mask to the input.

    Args:
      inputs: the inputs that should be randomly masked.
      deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
      rng: an optional PRNGKey used as the random key, if not specified, one
        will be generated using ``make_rng`` with the ``rng_collection`` name.

    Returns:
      The masked inputs reweighted to preserve mean.
    """
    deterministic = merge_param(
      'deterministic', self.deterministic, deterministic
    )

    if (self.rate == 0.0) or deterministic:
      return inputs

    # Prevent gradient NaNs in 1.0 edge-case.
    if self.rate == 1.0:
      return jnp.zeros_like(inputs, dtype=inputs.dtype)

    keep_prob = 1.0 - self.rate
    if rng is None:
      rng = self.make_rng(self.rng_collection)

    shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
    random_tensor = keep_prob + jax.random.uniform(rng, shape=shape, dtype=inputs.dtype)  #TODO jax.array(keep_prob)?
    output = jnp.divide(inputs, keep_prob) * jnp.floor(random_tensor)
    return output


def normalize_function(inputs, normalize_type, dtype, scale_factor=1., zoom=0.):
  if normalize_type == 'gelu':
    outputs = nn.gelu(inputs)
  elif normalize_type == 'softmax':
    outputs = nn.softmax(inputs, axis=-1) * inputs.shape[-1]
  elif normalize_type == 'l2_norm':
    norm = jnp.linalg.norm(inputs, ord=2, axis=-1, keepdims=True)
    if zoom != 0.:
      outputs = inputs / (norm + 1e-8) * jnp.sqrt(zoom)
    else:
      outputs = inputs / (norm + 1e-8) * jnp.sqrt(inputs.shape[-1])
  elif normalize_type == 'gelu_l2_norm':
    x = nn.gelu(inputs)
    norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    if zoom != 0.:
      outputs = x / (norm + 1e-8) * jnp.sqrt(zoom)
    else:
      outputs = x / (norm + 1e-8) * jnp.sqrt(inputs.shape[-1])
  elif normalize_type == 'l2_norm_gelu':
    norm = jnp.linalg.norm(inputs, ord=2, axis=-1, keepdims=True)
    if zoom != 0.:
      outputs = nn.gelu(inputs / (norm + 1e-5) * jnp.sqrt(zoom))
    else:
      outputs = nn.gelu(inputs / (norm + 1e-5) * jnp.sqrt(inputs.shape[-1]))
  elif normalize_type == 'gelu_layer_norm':
    outputs = nn.LayerNorm(use_bias=False, use_scale=False)(nn.gelu(inputs))
  else:
    raise NotImplementedError
  return outputs * scale_factor

# -----------------------------------------
# The Normalization Layer specific for TokenFormer
# -----------------------------------------


class TokenFormerLayerNorm(nn.Module):
  """TokenFormer Layer normalization operating on the last axis of the input data."""

  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  reductions_in_fp32: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    if self.reductions_in_fp32:
      x = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=[-1], keepdims=True)
    normed_inputs = (x - mean) * lax.rsqrt(var + self.epsilon)
    if self.reductions_in_fp32:
      normed_inputs = normed_inputs.astype(self.dtype)
    return normed_inputs



class MlpBlock(nn.Module):
  """TokenFormer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: computation data type for the dense layer.
    weight_dtype: weight data type for the dense layer.
    use_bias: whether to add bias in all feedforward layers.
    use_pre_norm: whether to add pre layer norm in mlp layers.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  intermediate_dim: int = 2048
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  normalize_type: str = 'l2_norm_gelu'
  ffn_scale_factor: float = 1.0
  ffn_bias: bool = False
  use_pre_norm: bool = False
  quant: Optional[Quant] = None
  zoom_dim: float = 0.

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.use_pre_norm:
      inputs = TokenFormerLayerNorm(
          name="mlp_layer_norm",
          dtype=cfg.dtype,
          reductions_in_fp32=False,
          epsilon=cfg.normalization_layer_epsilon,
      )(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    dense_name = "wi"
    x = DenseGeneral(
        self.intermediate_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "mlp"),
        name=dense_name,
        quant=self.quant,
        use_bias=self.ffn_bias,
    )(inputs)
    x = normalize_function(x, self.normalize_type, self.dtype, self.ffn_scale_factor, self.zoom_dim)
    x = checkpoint_name(x, "mlpwi")
    
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )  # Broadcast along length.
    x = nn.with_logical_constraint(x, (BATCH, "activation_length", "activation_mlp"))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        name="wo",
        quant=self.quant,
        use_bias=self.ffn_bias,
    )(x)

    output = checkpoint_name(output, "mlpwo")
    return output

# -----------------------------------------
# The Attention Layer specific for GPT3
# -----------------------------------------


class TokenFormerMultiHeadAttention(nn.Module):
  """Multi-head attention in TokenFormer.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    head_dim: dimension of each head.
    max_target_length: maximum length of output
    max_prefill_predict_length: size of the maximum prefill
    mesh: device mesh
    dtype: the dtype of the computation.
    dropout_rate: dropout rate
    kernel_init: initializer for the kernel of the Dense layers.
    float32_qk_product: bool, if True then compute logits via float32 qk_product to avoid
      numerical issues with bfloat16.
    float32_logits: bool, if True then cast logits to float32 before softmax to avoid
      numerical issues with bfloat16.
    fused_qkv: whether to fuse query, key and value into one projection.
    quant: Quant, stores quantization config, defaults to None implying no quantization.
    use_bias: whether to add bias in linear transformation.
  """

  config: Config
  num_heads: int
  head_dim: int
  max_target_length: int
  max_prefill_predict_length: int
  mesh: Mesh
  attention_kernel: str
  dtype: DType = jnp.float32
  weight_dtype: DType = jnp.float32
  dropout_rate: float = 0.0
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  float32_qk_product: bool = False  # computes logits in float32 for stability.
  float32_logits: bool = True  # cast logits in float32 for stability.
  quant: Optional[Quant] = None
  kv_quant: Optional[KVQuant] = None
  
  slot_num: Optional[int] = None
  normalize_type: str = 'l2_norm_gelu'
  qkv_scale_factor: float = 1.0
  proj_scale_factor: float = 1.0
  qkv_bias: bool = False
  proj_bias: bool = False
  zoom_dim: float = 0.

  query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  key_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  value_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      decoder_segment_ids: Array | None = None,
      *,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
      deterministic: bool = False,
  ):
    batch_size, seq_length, embed_dim = inputs_q.shape
    slot_num = self.slot_num or embed_dim
    qkv = DenseGeneral(
      features=(3, slot_num),
      axis=-1,
      kernel_init=self.kernel_init,
      kernel_axes=("embed", "qkv", "slot_num"),
      dtype=self.dtype,
      weight_dtype=self.weight_dtype,
      name="qkv_key_slot",
      quant=self.quant,
      use_bias=self.qkv_bias,
    )(inputs_q)
    
    qkv = normalize_function(qkv, self.normalize_type, self.dtype, self.qkv_scale_factor, self.zoom_dim)
    query, key, value = jnp.array_split(qkv, 3, axis=-2)
    query = DenseGeneral(
      features=embed_dim,
      axis=-1,
      kernel_init=self.kernel_init,
      kernel_axes=("slot_num", "embed"),
      dtype=self.dtype,
      weight_dtype=self.weight_dtype,
      name="q_value_slot",
      quant=self.quant,
      use_bias=self.qkv_bias,
    )(query)
    key = DenseGeneral(
      features=embed_dim,
      axis=-1,
      kernel_init=self.kernel_init,
      kernel_axes=("slot_num", "embed"),
      dtype=self.dtype,
      weight_dtype=self.weight_dtype,
      name="k_value_slot",
      quant=self.quant,
      use_bias=self.qkv_bias,
    )(key)
    value = DenseGeneral(
      features=embed_dim,
      axis=-1,
      kernel_init=self.kernel_init,
      kernel_axes=("slot_num", "embed"),
      dtype=self.dtype,
      weight_dtype=self.weight_dtype,
      name="v_value_slot",
      quant=self.quant,
      use_bias=self.qkv_bias,
    )(value)
    
    query = query.reshape(batch_size, seq_length, self.num_heads, -1)  # [Batch, SeqLen, Head, Dims]
    key = key.reshape(batch_size, seq_length, self.num_heads, -1)  # [Batch, SeqLen, Head, Dims]
    value = value.reshape(batch_size, seq_length, self.num_heads, -1)  # [Batch, SeqLen, Head, Dims]

    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query /= depth_scaling

    # annotate with sharding constraint.
    query = nn.with_logical_constraint(query, self.query_axis_names)
    query = checkpoint_name(query, "query_proj")
    key = nn.with_logical_constraint(key, self.key_axis_names)
    key = checkpoint_name(key, "key_proj")
    value = nn.with_logical_constraint(value, self.value_axis_names)
    value = checkpoint_name(value, "value_proj")
    
    # NOTE: the reason why loss is NaN is that Attention needs FP32
    query = jnp.asarray(query, jnp.float32)
    key = jnp.asarray(key, jnp.float32)
    value = jnp.asarray(value, jnp.float32)
    attention_op = AttentionOp(
        mesh=self.mesh,
        attention_kernel=self.attention_kernel,
        max_target_length=self.max_target_length,
        float32_qk_product=self.float32_qk_product,
        float32_logits=self.float32_logits,
        quant=self.quant,
        kv_quant=self.kv_quant,
        num_query_heads=self.num_heads,
        num_kv_heads=self.num_heads,
        dtype=self.dtype,
    )
    out = attention_op(query, key, value, decoder_segment_ids, model_mode)
    out = nn.with_logical_constraint(out, self.out_axis_names)
    out = out.reshape(batch_size, seq_length, embed_dim)
    out = out.astype(self.dtype)

    # apply output projection, output dim is set to the input dim.
    out = DenseGeneral(
        features=slot_num,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "slot_num"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="proj_key_slot",
        quant=self.quant,
        use_bias=self.proj_bias,
    )(out)
    out = normalize_function(out, self.normalize_type, self.dtype, self.proj_scale_factor, self.zoom_dim)
    out = DenseGeneral(
      features=embed_dim,
      axis=-1,
      kernel_init=self.kernel_init,
      kernel_axes=("slot_num", "embed"),
      dtype=self.dtype,
      weight_dtype=self.weight_dtype,
      name="proj_value_slot",
      quant=self.quant,
      use_bias=self.proj_bias,
    )(out)
    out = checkpoint_name(out, "out_proj")
    out = nn.Dropout(rate=self.dropout_rate)(out, deterministic) 
    return out


# -----------------------------------------
# The Decoder Layer specific for TokenFormer
# -----------------------------------------


class TokenFormerDecoderLayer(nn.Module):
  """TokenFormer decoder layer that attends to the encoder."""

  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
  ):
    cfg = self.config
    mesh = self.mesh
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))

    lnx_layer_norm = TokenFormerLayerNorm(
        dtype=cfg.dtype,
        name="pre_self_attention_norm",
        epsilon=cfg.normalization_layer_epsilon,
        reductions_in_fp32=False,
    )
    lnx = lnx_layer_norm(inputs)

    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

    # Self-attention block
    assert (
        cfg.num_query_heads == cfg.num_kv_heads
    ), f"{cfg.num_query_heads=} should be the same as {cfg.num_kv_heads=} in gpt3"
    attention_layer = TokenFormerMultiHeadAttention(
        config=cfg,
        num_heads=cfg.num_query_heads,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        mesh=mesh,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        
        slot_num=cfg.slot_num,
        normalize_type=cfg.normalize_type,
        qkv_scale_factor=cfg.qkv_scale_factor,
        proj_scale_factor=cfg.proj_scale_factor,
        qkv_bias=cfg.qkv_bias,
        proj_bias=cfg.proj_bias,
        zoom_dim=cfg.zoom_dim,
    )

    attention_lnx = attention_layer(
        lnx, decoder_segment_ids=decoder_segment_ids, model_mode=model_mode, deterministic=deterministic
    )

    attention_lnx = nn.with_logical_constraint(attention_lnx, ("activation_batch", "activation_length", "activation_embed"))

    attention_lnx = DropPath(rate=cfg.drop_path)(attention_lnx, deterministic)

    attention_lnx += inputs

    # MLP block.
    mlp_lnx = MlpBlock(
        intermediate_dim=4 * cfg.slot_num,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        normalize_type=cfg.normalize_type,
        ffn_scale_factor=cfg.ffn_scale_factor,
        ffn_bias=cfg.ffn_bias,
        use_pre_norm=True,
        config=cfg,
        quant=self.quant,
        zoom_dim=cfg.zoom_dim * 4,
    )(attention_lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    mlp_lnx = DropPath(rate=cfg.drop_path)(mlp_lnx, deterministic)
    
    layer_output = attention_lnx + mlp_lnx

    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
