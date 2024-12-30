from typing import Any, Optional, Sequence
import torch
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

class SliceUpdateKeyValueCache(Cache):
    """Helper class for in-place slice updating key/value caches."""
    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create key/value cache of shape:
        (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k: torch.Tensor = torch.zeros(shape, dtype=dtype)
        self.v: torch.Tensor = torch.zeros(shape, dtype=dtype)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update key / value cache tensors for slice [begin, end).
        
        Args:
            k_state: Key state tensor
            v_state: Value state tensor
            layer_idx: Index of the layer to update
            cache_kwargs: Additional cache arguments
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updated key and value states
            
        Raises:
            ValueError: If cache_position is missing or invalid
            IndexError: If layer_idx is out of bounds
        """
        if cache_kwargs is None:
            raise ValueError("cache_kwargs cannot be None")
            
        position = cache_kwargs.get("cache_position")
        if position is None:
            raise ValueError("cache_position required to update cache")
            
        if not 0 <= layer_idx < self.k.shape[0]:
            raise IndexError(f"Layer index {layer_idx} out of bounds [0, {self.k.shape[0]})")
            
        begin, end = self.past_seen_tokens, self.past_seen_tokens + position
        if end > self.k.shape[3]:  # context_size dimension
            raise ValueError(f"Cache overflow: {end} exceeds context size {self.k.shape[3]}")
        self.k[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        k_state = self.k[layer_idx, :, :, :end, :]
        v_state = self.v[layer_idx, :, :, :end, :]
        return k_state, v_state

    def get_seq_length(self, _: int = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens

class KvCacheStateLlamaForCausalLM(torch.nn.Module):
    """Model wrapper to swap cache implementation and register as buffers."""
    def __init__(
        self,
        model_path: str, *,
        batch_size: int = 1, 
        context_size: int = 4096,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the model with KV cache.
        
        Args:
            model_path: Path to the pretrained model
            batch_size: Batch size for inference
            context_size: Maximum context window size
            device: Device to place the model on
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LlamaForCausalLM.from_pretrained(model_path).to(self.device)
        config: LlamaConfig = self.model.config
        self.kv_cache_shape: tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            context_size,
            config.hidden_size // config.num_attention_heads,
        )
        # Register KV cache buffers to be recognized as Core ML states
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k)
        self.register_buffer("valueCache", self.kv_cache.v)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits
