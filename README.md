# DropKV

[**DropKV: Decoupling Residual-Output Perturbation for
Near-Optimal KV-Cache
Eviction**](https://openreview.net/pdf?id=UVbaJoD3Qa), has been accepted to
**NeurIPS 2026 as a poster**.

DropKV is an attention-output-sensitive KV-cache eviction method for LLM
inference. During prefill, it scores KV positions by their estimated effect on
the attention output and retains the highest-scoring fraction. Decode then
continues with the compressed cache.

## What is included

- `dropkv_cache.py`: standalone `DropKVCache` and chunked-prefill helper.
- `dropkv_triton.py`: Triton kernels and a PyTorch reference implementation.
- `patches/dropkv_transformers.patch`: integration for a pinned Transformers
  revision, including Llama and Qwen2 query-state plumbing.
- `tests/test_dropkv_triton.py`: kernel, cache, chunked-prefill, and guard tests.

Experimental outputs, private cluster paths, unrelated eviction methods, model weights, and vendored third-party repositories are intentionally excluded.

## Requirements

- Linux with an NVIDIA GPU for the Triton path
- Python 3.9 or newer
- PyTorch 2.6 or newer
- Triton 3.0 or newer (optional; PyTorch fallback is available)
- A patched checkout of Hugging Face Transformers for `generate()` and `lm-eval` integration

The integration patch is pinned to Transformers commit `fd6bc380c8854a370fbc9f68a157895d84dce7d7` (the repository identified it as `v5.3.0.dev0`). Do not assume that it applies to another revision.

## Installation

The setup script creates a Conda environment, clones the pinned Transformers revision, applies the DropKV patch, and installs the local project:

```bash
bash setup.sh
conda activate dropkv
```

For an existing environment, install the local modules first, then apply the patch to an exact checkout of the pinned revision:

```bash
python -m pip install -e '.[triton,test]'

git clone https://github.com/huggingface/transformers.git transformers_src
git -C transformers_src checkout fd6bc380c8854a370fbc9f68a157895d84dce7d7
git -C transformers_src apply ../patches/dropkv_transformers.patch
python -m pip install -e ./transformers_src
```

## Usage

### Standalone cache

The model attention implementation must pass `query_states` in `cache_kwargs`; the supplied Transformers patch does this for Llama and Qwen2.

```python
from dropkv_cache import DropKVCache

cache = DropKVCache(
    keep_ratio=0.1,
    window_size=8,
    kernel_size=11,
    use_triton=True,
)

outputs = model(
    input_ids,
    past_key_values=cache,
    use_cache=True,
)
```

### Scoring API

```python
from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton

scores = dropkv_scores_triton(Q, K, V, window_size=8)
reference = dropkv_scores_reference(Q, K, V, window_size=8)
```

`Q` has shape `[B, Hq, Lq, D]`; `K` and `V` have shape
`[B, Hkv, Lk, D]`. Grouped-query attention is supported when `Hq` is divisible
by `Hkv`, and chunked prefill is supported when `Lq <= Lk` and the query is the
tail-aligned suffix.

### lm-eval

After running `setup.sh`:

```bash
bash run_eval.sh \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --task ruler \
  --keep-ratio 0.1 \
  --output ./results
```

Paths may also be configured by copying `.env.example` to `.env`.

## Tests and validation

```bash
python -m pytest tests/test_dropkv_triton.py -v
```

Most tests require CUDA. Several integration tests also download
`Qwen/Qwen2-0.5B-Instruct` and require the patched Transformers checkout.

Before testing a modified patch, verify that it still applies cleanly:

```bash
git -C transformers_src reset --hard fd6bc380c8854a370fbc9f68a157895d84dce7d7
git -C transformers_src apply --check ../patches/dropkv_transformers.patch
```

The `reset --hard` command above is only for a disposable Transformers clone;
do not run it in a checkout containing work you need to keep.

## License

DropKV is released under the [Apache License 2.0](LICENSE). Portions adapted
from Hugging Face Transformers remain under Apache-2.0; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## Citation

If you find DropKV useful in your research, please cite:

```bibtex
@inproceedings{zhang2026dropkv,
  title     = {DropKV: Decoupling Residual-Output Perturbation for Near-Optimal KV-Cache Eviction},
  author    = {Zhang, Aozhong and Gurses, Selcuk and Deng, Yanxia and Wang, Naigang and Liu, Chi-Chun and Wertheimer, Davis and Liu, Derrick and Li, Xin and Yang, Zi and Ye, Felix X.-F. and Yin, Penghang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
