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

"""Qwen3 model configuration definitions."""

import logging

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

ALLOWED_LAYER_TYPES = (
    "full_attention",
    "sliding_attention",
    "chunked_attention",
)


def layer_type_validation(layer_types: list[str]):
    """Check that every entry in `layer_types` is allowed."""
    if not all(layer_type in ALLOWED_LAYER_TYPES for layer_type in layer_types):
        raise ValueError(f"The `layer_types` entries must be in {ALLOWED_LAYER_TYPES}")


logger = logging.getLogger(__name__)


class Qwen3Config(PretrainedConfig):
    r"""
    Configuration class for [`Qwen3Model`]. The supplied arguments define the model
    architecture. The defaults produce a configuration similar to
    [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

    Configuration objects inherit from [`PretrainedConfig`] and control model
    outputs. See the [`PretrainedConfig`] documentation for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3 model. Defines the number of tokens that
            can be represented by the `input_ids` passed to [`Qwen3Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in each Transformer decoder layer.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key/value heads used for grouped-query attention (GQA). A
            value equal to `num_attention_heads` selects multi-head attention
            (MHA), while `1` selects multi-query attention (MQA). When converting
            an MHA checkpoint to GQA, construct each key/value head by mean-pooling
            the original heads in that group. See the [GQA
            paper](https://arxiv.org/pdf/2305.13245.pdf) for details.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            Nonlinear activation function (function or string) used in the decoder.
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
        rope_scaling (`Dict`, *optional*):
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
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value, and output projection
            layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding-window attention (SWA) window size.
        max_window_layers (`int`, *optional*, defaults to 28):
            Number of lower layers that use SWA; the remaining upper layers use
            full attention.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen3Model, Qwen3Config

    >>> # Initialize a Qwen3-style configuration
    >>> configuration = Qwen3Config()

    >>> # Initialize a model from the Qwen3-8B-style configuration
    >>> model = Qwen3Model(configuration)

    >>> # Access the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # Backward compatibility: rename the legacy ``type`` field to ``rope_type``.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                (
                    "sliding_attention"
                    if self.sliding_window is not None and i >= self.max_window_layers
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3Config"]
