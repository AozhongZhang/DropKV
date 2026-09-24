from collections.abc import Iterable
from typing import Any, Optional

import logging
import math

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    is_compileable = False

    def __init__(self):
        super().__init__()

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        raise NotImplementedError("Make sure to implement `get_max_cache_shape` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx].numel():
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        logger.warning(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, 0


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self, _distributed_cache_data: Optional[Iterable] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        # `_distributed_cache_data` was originally added for compatibility with `torch.distributed` (DDP). See #36121
        # and #36373 for more information. In a nutshell, it is `map(gather_map, zip(*caches))`, i.e. each item in the
        # iterable contains the key and value states for a layer gathered across replicas by torch.distributed
        # (shape=[global batch size, num_heads, seq_len, head_dim]).
        # WARNING: `_distributed_cache_data` must be the first argument in `__init__`, otherwise we'll break
        # compatibility. The name of the argument doesn't matter.
        if _distributed_cache_data is not None:
            for key_states, value_states in _distributed_cache_data:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[tuple[tuple[torch.FloatTensor, torch.FloatTensor]]] = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(None, key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx].numel():
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> list["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: list["DynamicCache"], **kwargs) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls(**kwargs)
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx].numel()]
            value_cache = [current.value_cache[idx] for current in splits if current.value_cache[idx].numel()]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(None, layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class DropKVCache(DynamicCache):

    def __init__(
        self,
        keep_ratio: float = 0.1,
        window_size: int = 8,
        kernel_size: int = 11,
        use_triton: bool = True,
    ) -> None:
        super().__init__()

        if not 0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size = window_size
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.kernel_size = kernel_size
        self.use_triton = use_triton

        # Prefill mode state
        # "single_pass"        — default: score + evict in one forward
        # "accumulate"         — chunked prefill: just append KV (no eviction)
        # "evict_accumulated"  — chunked prefill: concat last chunk, then score + evict
        self._prefill_mode = "single_pass"
        # When True, evict_accumulated uses _seen_tokens for keep_len target
        # (budget/every_chunk strategies). When False, uses current KV length.
        self._evict_use_seen_tokens = False
        self._budget_max_kv_tokens: Optional[int] = None

    def batch_split(self, full_batch_size: int, split_size: int) -> list["DynamicCache"]:
        raise NotImplementedError(
            "DropKVCache does not support batch_split (beam search). "
            "Use DynamicCache for beam search or implement DropKV-aware splitting."
        )

    @classmethod
    def from_batch_splits(cls, splits: list["DynamicCache"], **kwargs) -> "DynamicCache":
        raise NotImplementedError(
            "DropKVCache does not support from_batch_splits (beam search). "
            "Use DynamicCache for beam search or implement DropKV-aware merging."
        )

    def _compute_keep_indices(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        total_tokens: Optional[int] = None,
        max_keep_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute DropKV eviction scores and return sorted top-k indices.

        Args:
            query_states: [bsz, num_query_heads, q_len, head_dim]
                q_len may be shorter than kv_len (chunked prefill). Must be
                tail-aligned: last query position corresponds to last key position.
            key_states:   [bsz, num_kv_heads, kv_len, head_dim]
            value_states: [bsz, num_kv_heads, kv_len, head_dim]
            total_tokens: If set, compute keep_len from this value instead
                of seq_len.  Used by budget-based eviction where keep_len
                should grow with total tokens seen, not current KV size.
            max_keep_len: Hard upper bound on keep_len.  Prevents retained
                KV from exceeding the budget when keep_ratio * total_tokens
                would otherwise surpass max_kv_tokens.

        Returns:
            keep_idx: [bsz, num_kv_heads, keep_len] sorted indices to keep.
        """
        bsz, num_kv_heads, seq_len, head_dim = key_states.shape
        num_query_heads = query_states.shape[1]

        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"Query heads {num_query_heads} must be divisible by KV heads {num_kv_heads}")
        num_key_value_groups = num_query_heads // num_kv_heads

        # Compute scores (Triton or PyTorch fallback)
        _use_triton = self.use_triton
        if _use_triton:
            try:
                from dropkv_triton import dropkv_scores_triton
                scores = dropkv_scores_triton(
                    query_states,
                    key_states.contiguous(),
                    value_states.contiguous(),
                    window_size=self.window_size,
                )
            except (ImportError, AssertionError):
                _use_triton = False
        if not _use_triton:
            # GQA-aware scoring: reshape Q to [B, Hkv, g*W, D] and compute
            # attention directly against K/V at [B, Hkv, L, D].
            # This avoids repeat_kv (~1 GB at L=65K) and the large WT_y
            # intermediate (~1 GB), saving ~2 GB total.
            g = num_key_value_groups
            W = min(self.window_size, query_states.shape[2])
            M = g * W  # total query rows per KV head

            # Reshape Q window: [B, Hq, W, D] -> [B, Hkv, g*W, D]
            Q_w = query_states[..., -W:, :].reshape(bsz, num_kv_heads, M, head_dim)

            # Hoist fp32 casts (each used 2+ times below)
            K_f = key_states.float()
            V_f = value_states.float()

            # Attention logits: [B, Hkv, M, L]
            attn_logits = torch.matmul(
                Q_w.float(), K_f.transpose(-1, -2)
            ) / math.sqrt(head_dim)

            # Causal mask: row m has causal bound L - W + (m % W)
            offs_m = torch.arange(M, device=key_states.device)
            causal_bounds = seq_len - W + (offs_m % W)  # [M]
            col_idx = torch.arange(seq_len, device=key_states.device)  # [L]
            attn_logits.masked_fill_(
                col_idx[None, None, None, :] > causal_bounds[None, None, :, None],
                float("-inf"),
            )

            p = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # y = P @ V: [B, Hkv, M, D] — bf16 (matching original path)
            y = torch.matmul(p, value_states)

            # W_mat = (p / (1 - p + eps))^2 — fp32
            eps = 1e-6
            p_f = p.float()
            W_mat = (p_f / (1.0 - p_f + eps)) ** 2  # [B, Hkv, M, L]

            # Term A: sum_m(W) * ||v||^2
            W_sum = W_mat.sum(dim=-2)  # [B, Hkv, L]
            v_norm2 = (V_f ** 2).sum(dim=-1)  # [B, Hkv, L]
            termA = W_sum * v_norm2

            # Term B: 2 * sum_m(W_mj * <v_j, y_m>)
            # V @ y^T -> [B, Hkv, L, M] (M=32, much smaller than [B,Hq,L,D])
            V_dot_y = torch.matmul(
                V_f, y.float().transpose(-1, -2)
            )  # [B, Hkv, L, M]
            termB = 2.0 * (W_mat.transpose(-1, -2) * V_dot_y).sum(dim=-1)

            # Term C: sum_m(W_mj * ||y_m||^2)
            y_norm2 = (y.float() ** 2).sum(dim=-1)  # [B, Hkv, M]
            termC = torch.matmul(
                W_mat.transpose(-1, -2), y_norm2.unsqueeze(-1)
            ).squeeze(-1)  # [B, Hkv, L]

            # Average over g query groups
            scores = (termA - termB + termC) / g  # [B, Hkv, L]

        # Smooth and protect window tokens
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        big = scores.amax(dim=-1, keepdim=True) + 1.0
        if self.window_size > 0:
            scores[:, :, -self.window_size:] = big

        # Top-k selection
        ratio_base = total_tokens if total_tokens is not None else seq_len
        keep_len = min(seq_len, max(int(self.keep_ratio * ratio_base), self.window_size, 1))
        if max_keep_len is not None:
            keep_len = min(keep_len, max_keep_len)
        keep_idx = torch.topk(scores, k=keep_len, dim=-1, largest=True).indices
        keep_idx, _ = torch.sort(keep_idx, dim=-1)

        return keep_idx

    @staticmethod
    def _gather_compressed(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        keep_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather compressed KV using pre-computed indices."""
        head_dim = key_states.shape[-1]
        keep_idx_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        return (
            torch.gather(key_states, dim=2, index=keep_idx_expanded),
            torch.gather(value_states, dim=2, index=keep_idx_expanded),
        )

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compatibility: detect HF 4-arg call signature
        # HF calls:  update(key, value, layer_idx:int, cache_kwargs:dict)
        # Our sig:   update(query, key, value, layer_idx:int, cache_kwargs)
        # When HF calls us, value_states receives the int, layer_idx receives the dict
        if isinstance(value_states, int) and isinstance(layer_idx, dict):
            cache_kwargs = layer_idx
            layer_idx = value_states
            value_states = key_states
            key_states = query_states
            query_states = cache_kwargs.get("query_states") if cache_kwargs else None
            if query_states is None:
                logger.warning(
                    "DropKVCache.update() called without query_states — "
                    "eviction disabled for this layer. Patch your transformers "
                    "to pass query_states in cache_kwargs."
                )

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # --- Phase 1: Prefill  ---

        if len(self.key_cache) < layer_idx:
            raise ValueError("Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:

            # No query_states: store full KV without eviction (e.g. from_legacy_cache)
            if query_states is None:
                if self._prefill_mode != "single_pass":
                    logger.warning(
                        "query_states is None at layer %d — "
                        "storing full KV without eviction.", layer_idx
                    )
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                return key_states, value_states

            bsz, num_kv_heads, seq_len, head_dim = key_states.shape

            _valid_modes = ("single_pass", "accumulate", "evict_accumulated")
            if self._prefill_mode not in _valid_modes:
                raise ValueError(f"Invalid _prefill_mode: {self._prefill_mode!r}")

            if self._prefill_mode == "accumulate":
                # Chunked prefill: store KV without eviction (first chunk)
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                return key_states, value_states

            if seq_len < self.window_size:
                # Sequence too short for scoring — skip eviction
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                return key_states, value_states

            # single_pass or evict_accumulated: score + evict
            total_tok = self._seen_tokens if self._evict_use_seen_tokens else None
            keep_idx = self._compute_keep_indices(
                query_states, key_states, value_states,
                total_tokens=total_tok,
                max_keep_len=self._budget_max_kv_tokens,
            )
            key_c, value_c = self._gather_compressed(key_states, value_states, keep_idx)
            self.key_cache.append(key_c)
            self.value_cache.append(value_c)
            return key_states, value_states

        # --- Phase 2: Decoding / Chunked prefill continuation ---
        else:
            if self._prefill_mode == "accumulate":
                # Chunked prefill: concatenate without eviction
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            elif self._prefill_mode == "evict_accumulated":
                # Chunked prefill eviction: concat, then score and evict
                full_key = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                full_value = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

                if query_states is not None and full_key.shape[2] >= self.window_size:
                    total_tok = self._seen_tokens if self._evict_use_seen_tokens else None
                    keep_idx = self._compute_keep_indices(
                        query_states, full_key, full_value,
                        total_tokens=total_tok,
                        max_keep_len=self._budget_max_kv_tokens,
                    )
                    key_c, value_c = self._gather_compressed(
                        full_key, full_value, keep_idx
                    )
                    self.key_cache[layer_idx] = key_c
                    self.value_cache[layer_idx] = value_c
                else:
                    self.key_cache[layer_idx] = full_key
                    self.value_cache[layer_idx] = full_value

                # Return full KV for this layer's attention (same as single_pass)
                return full_key, full_value

            elif self._prefill_mode == "single_pass":
                # Standard decode: concatenate
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            else:
                raise ValueError(
                    f"Invalid _prefill_mode {self._prefill_mode!r} in decode path"
                )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the logical sequence length (total tokens seen) for correct RoPE positions.

        After eviction the stored KV is shorter than the true sequence length.
        This returns _seen_tokens (the logical length) so that decode tokens
        receive correct positional embeddings.
        """
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0) -> tuple[int, int]:
        """Return (kv_length, kv_offset) matching the stored KV dimensions.

        The attention mask must match the actual stored tensor size, not the
        logical length, so we use the stored key tensor length here.
        """
        query_length = cache_position.shape[0]
        is_empty = (
            len(self.key_cache) == 0
            or len(self.key_cache) <= layer_idx
            or not self.key_cache[layer_idx].numel()
        )
        stored_len = 0 if is_empty else self.key_cache[layer_idx].shape[-2]
        return stored_len + query_length, 0


def prefill_chunked(
    model,
    input_ids,
    cache,
    chunk_size=4096,
    evict_strategy="last_chunk",
    max_kv_tokens=32768,
    **forward_kwargs,
):
    """Chunked prefill to reduce per-layer activation peak memory.

    Splits the prompt into chunks of ``chunk_size`` tokens.  Each chunk is
    processed in a separate ``model.forward()`` call, keeping per-layer
    activations proportional to ``B * chunk_size`` instead of ``B * L``.

    The ``evict_strategy`` controls when DropKV eviction happens:

    - ``"last_chunk"`` (default): accumulate full KV, evict only on the last
      chunk.  Near-exact match to single-pass quality.  (If the final chunk
      has fewer tokens than ``window_size``, scoring uses fewer query
      positions than single-pass, causing minor index differences.)
    - ``"budget"``: accumulate KV until it exceeds ``max_kv_tokens``, then
      evict to ``keep_ratio * total_seen_tokens``.  Bounds GPU KV memory
      with a few eviction events.
    - ``"every_chunk"``: evict after every chunk.  Lowest memory but most
      approximate (equivalent to ``budget`` with ``max_kv_tokens=0``).

    Args:
        model: HuggingFace causal LM (in eval mode, on GPU).
        input_ids: ``[B, L]`` prompt token ids on the model's device.
        cache: An ``DropKVCache`` instance (will be mutated).
        chunk_size: Max tokens per forward pass.  Must be >= ``cache.window_size``.
        evict_strategy: ``"last_chunk"`` | ``"budget"`` | ``"every_chunk"``.
        max_kv_tokens: KV budget for ``"budget"`` strategy.  Eviction triggers
            when accumulated KV exceeds this.  Also caps keep_len so retained
            KV never exceeds the budget.  Must be >= ``window_size``.
            Note: this bounds the *stored* KV between chunks, not peak
            transient memory (eviction temporarily materializes full KV
            via ``torch.cat``).  Ignored for other strategies.
        **forward_kwargs: Extra kwargs passed to ``model.forward()``.
            Note: ``attention_mask``, ``position_ids``, and ``cache_position``
            are NOT supported — the model computes these internally per chunk.

    Returns:
        Model output from the last chunk (with compressed cache attached).
    """
    if not isinstance(cache, DropKVCache):
        raise TypeError(
            f"cache must be DropKVCache, got {type(cache).__name__}"
        )
    if len(cache.key_cache) > 0:
        raise ValueError(
            "prefill_chunked() requires a fresh (empty) cache. "
            "Create a new DropKVCache instance."
        )
    if chunk_size < cache.window_size:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be >= window_size ({cache.window_size})"
        )
    _seq_kwargs = {"attention_mask", "position_ids", "cache_position"}
    bad_kwargs = _seq_kwargs & forward_kwargs.keys()
    if bad_kwargs:
        raise ValueError(
            f"prefill_chunked() does not support sequence-shaped kwargs "
            f"{bad_kwargs} — the model computes these internally per chunk. "
            f"Remove them from forward_kwargs."
        )

    valid_strategies = ("last_chunk", "budget", "every_chunk")
    if evict_strategy not in valid_strategies:
        raise ValueError(
            f"evict_strategy must be one of {valid_strategies}, got {evict_strategy!r}"
        )
    if evict_strategy == "budget":
        if max_kv_tokens is None:
            raise ValueError(
                "max_kv_tokens is required when evict_strategy='budget'"
            )
        if max_kv_tokens < cache.window_size:
            raise ValueError(
                f"max_kv_tokens ({max_kv_tokens}) must be >= window_size "
                f"({cache.window_size}) to preserve protected tail tokens"
            )

    B, L = input_ids.shape

    # If prompt fits in one chunk AND strategy doesn't need post-hoc
    # enforcement (budget/every_chunk need the loop to cap keep_len),
    # just do normal single-pass.
    if L <= chunk_size and evict_strategy == "last_chunk":
        with torch.no_grad():
            return model(
                input_ids, past_key_values=cache, use_cache=True,
                **forward_kwargs,
            )

    chunk_starts = list(range(0, L, chunk_size))

    # Budget/every_chunk strategies use _seen_tokens for keep_len
    use_seen = evict_strategy in ("budget", "every_chunk")

    try:
        cache._evict_use_seen_tokens = use_seen
        if evict_strategy == "budget":
            cache._budget_max_kv_tokens = max_kv_tokens

        with torch.no_grad():
            output = None
            for i, start in enumerate(chunk_starts):
                end = min(start + chunk_size, L)
                chunk = input_ids[:, start:end]
                is_last = (i == len(chunk_starts) - 1)

                if evict_strategy == "last_chunk":
                    if is_last:
                        cache._prefill_mode = "evict_accumulated"
                    else:
                        cache._prefill_mode = "accumulate"

                elif evict_strategy == "every_chunk":
                    if i == 0:
                        # First chunk: prefill branch handles single_pass
                        cache._prefill_mode = "single_pass"
                    else:
                        cache._prefill_mode = "evict_accumulated"

                elif evict_strategy == "budget":
                    if is_last:
                        # Last chunk: always evict
                        cache._prefill_mode = "evict_accumulated"
                    else:
                        # Check if accumulated KV would exceed budget
                        stored = cache.key_cache[0].shape[2] if cache.key_cache else 0
                        would_have = stored + (end - start)
                        if would_have > max_kv_tokens:
                            cache._prefill_mode = "evict_accumulated"
                        else:
                            cache._prefill_mode = "accumulate"

                output = model(
                    chunk,
                    past_key_values=cache,
                    use_cache=True,
                    **forward_kwargs,
                )

            return output
    except BaseException:
        # On failure, release any partially-built cache to free GPU memory
        cache.key_cache.clear()
        cache.value_cache.clear()
        cache._seen_tokens = 0
        raise
    finally:
        cache._prefill_mode = "single_pass"
        cache._evict_use_seen_tokens = False
        cache._budget_max_kv_tokens = None
