# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2025 The Qwen team, Alibaba Group
# SPDX-FileCopyrightText: Copyright 2025 HuggingFace Inc. team
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Qwen3.5 text model configuration definitions."""

import enum

import numpy as np
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

from tokenspeed.runtime.distributed.utils import divide
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
)
from tokenspeed.runtime.utils.env import envs

logger = logging.get_logger(__name__)


#  HybridLayerType
class HybridLayerType(enum.Enum):
    full_attention = "attention"
    swa_attention = "swa_attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"


class Qwen3_5BaseTextConfig(PretrainedConfig):
    r"""
    Shared text configuration base used by Qwen3.5 dense and MoE configs.

    Configuration objects inherit from [`PretrainedConfig`] and control model
    outputs. See the [`PretrainedConfig`] documentation for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the model. Defines the number of tokens that can be
            represented by `input_ids`.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in each Transformer decoder layer.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            Number of key/value heads used for grouped-query attention (GQA). A
            value equal to `num_attention_heads` selects multi-head attention
            (MHA), while `1` selects multi-query attention (MQA). When converting
            an MHA checkpoint to GQA, construct each key/value head by mean-pooling
            the original heads in that group. See the [GQA
            paper](https://arxiv.org/pdf/2305.13245.pdf) for details.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Nonlinear activation function used in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length supported by the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer used for all
            weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model returns cached key and value states. This option is
            relevant only when `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_parameters (`Dict`, *optional*):
            Scaling configuration for the RoPE embeddings. Update this value when
            applying a new RoPE type for sequences longer than
            `max_position_embeddings`.
            Expected contents:
                `rope_type` (`str`):
                    RoPE variant to use. Supported values are `default`, `linear`,
                    `dynamic`, `yarn`, `longrope`, and `llama3`; `default` selects
                    the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all RoPE types except `default`. For most scaling
                    types, a factor of `x` supports sequences up to `x` times the
                    original maximum pretraining length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with `dynamic`, `longrope`, and `llama3`. Original
                    maximum position embeddings used during pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with `yarn` and `longrope`. Scaling factor applied to the
                    attention computation. If unset, the implementation derives a
                    recommended value from `factor`.
                `beta_fast` (`float`, *optional*):
                    Used only with `yarn`. Extrapolation boundary in the linear
                    ramp function. Defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Used only with `yarn`. Interpolation boundary in the linear
                    ramp function. Defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Used only with `longrope`. Scaling factors applied to contexts
                    shorter than `original_max_position_embeddings`. The list
                    length must equal the hidden size divided by twice the number
                    of attention heads.
                `long_factor` (`List[float]`, *optional*):
                    Used only with `longrope`. Scaling factors applied to contexts
                    longer than `original_max_position_embeddings`. The list
                    length must equal the hidden size divided by twice the number
                    of attention heads.
                `low_freq_factor` (`float`, *optional*):
                    Used only with `llama3`. Scaling factor applied to low-frequency
                    components of RoPE.
                `high_freq_factor` (`float`, *optional*):
                    Used only with `llama3`. Scaling factor applied to high-frequency
                    components of RoPE.
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            Fraction of each query and key to which rotary embeddings are applied.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value, and output projection
            layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 256):
            Dimension of each attention head.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the convolution used in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each key head in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each value head in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads used in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads used in linear attention layers.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the routed expert.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the shared expert.
        num_experts_per_tok (`int`, *optional*, defaults to 10):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 512):
            Number of routed experts.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the top-k probabilities.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether the model returns router logits and the auxiliary loss,
            including load-balancing and router z-loss terms.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Auxiliary-loss coefficient.
        mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
            Indices of layers that use dense MLPs rather than sparse MoE blocks.
            The list contains layer indices from 0 to `num_layers - 1`.
            If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.
        layer_types (`list[str]`, *optional*, defaults to `None`):
            Types of each layer (attention or linear).

    ```python
    >>> from transformers import Qwen3_5TextModel, Qwen3_5BaseTextConfig

    >>> # Initialize a Qwen3.5 text configuration
    >>> configuration =  Qwen3_5BaseTextConfig()

    >>> # Initialize a model from the Qwen3.5 text-80B-A3B-style configuration
    >>> model = Qwen3_5TextModel(configuration)

    >>> # Access the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "qwen3_5_text_base"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_parameters=None,
        partial_rotary_factor=0.25,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=10,
        num_experts=512,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=[],
        layer_types=None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        rope_config_validation(self)

        # linear attention (gdn now part)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers

    @property
    def layers_block_type(self):
        layer_type_list = []

        for layer_index in range(self.num_hidden_layers):
            if (layer_index + 1) % self.full_attention_interval == 0:
                layer_type_list.append(HybridLayerType.full_attention.value)
            else:
                layer_type_list.append(HybridLayerType.linear_attention.value)

        return layer_type_list

    @property
    def layer_types(self):
        """Per-layer paged-cache labels: "full_attention" / "linear_attention".

        Same interleaving as ``layers_block_type``, translated to the label
        vocabulary of ``paged_cache_spec`` (``HybridLayerType.full_attention``
        serializes as the checkpoint's "attention", which the KV-cache layer
        has no retention entry for). A property rather than an ``__init__``
        attribute because NextN drafts overwrite ``full_attention_interval``
        after construction (models/qwen3_5_nextn.py).
        """
        return [
            (
                FULL_ATTENTION
                if layer_type == HybridLayerType.full_attention.value
                else layer_type
            )
            for layer_type in self.layers_block_type
        ]

    @property
    def linear_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.linear_attention.value
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.full_attention.value
        ]

    @property
    def mamba2_cache_params(self):
        """Return per-layer cache shapes using the kernels' native layouts.

        The temporal/SSM state shape is K-last ``[Hv, V, K]``, matching the
        GDN decode, MTP, and chunk-prefill kernel boundary directly.
        """
        # Imported lazily to avoid config/env import cycles during module initialization.
        from tokenspeed.runtime.utils.env import global_server_args_dict

        self.mapping = global_server_args_dict["mapping"]
        attn_tp_size = self.mapping.attn.tp_size

        conv_dim = (
            self.linear_key_head_dim * self.linear_num_key_heads * 2
            + self.linear_value_head_dim * self.linear_num_value_heads
        )
        conv_state_shape = (
            divide(conv_dim, attn_tp_size),
            self.linear_conv_kernel_dim - 1,
        )

        temporal_state_shape = (
            divide(self.linear_num_value_heads, attn_tp_size),
            self.linear_value_head_dim,
            self.linear_key_head_dim,
        )
        conv_dtype = torch.bfloat16
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        ssm_dtype = dtype_map[envs.TOKENSPEED_MAMBA_SSM_DTYPE.get()]
        mamba_layers = self.linear_layer_ids
        return (
            conv_state_shape,
            temporal_state_shape,
            conv_dtype,
            ssm_dtype,
            mamba_layers,
        )

    @property
    def mamba_cache_per_req(self):
        conv_state_shape, temporal_state_shape, conv_dtype, ssm_dtype, mamba_layers = (
            self.mamba2_cache_params
        )
        mamba_layers_len = len(mamba_layers)

        return (
            int(np.prod(conv_state_shape)) * conv_dtype.itemsize
            + int(np.prod(temporal_state_shape)) * ssm_dtype.itemsize
        ) * mamba_layers_len
