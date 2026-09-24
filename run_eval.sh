#!/usr/bin/env bash
# Run lm-eval with DropKV cache eviction.
#
# Usage:
#   bash run_eval.sh --model /path/to/model            # minimal
#   bash run_eval.sh --model meta-llama/Llama-3.1-8B-Instruct --task ruler --limit 128
#   bash run_eval.sh --model /path/to/model --keep-ratio 0.2 --window-size 16
#
# Paths can also be set via .env file or environment variables:
#   MODEL_PATH, OUTPUT_DIR
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

# Defaults (overridable via env or CLI)
MODEL="${MODEL_PATH:-}"
OUTPUT="${OUTPUT_DIR:-./results}"
TASK="ruler"
LIMIT=""
BATCH_SIZE="auto"
KEEP_RATIO="0.1"
WINDOW_SIZE="8"
KERNEL_SIZE="11"
USE_TRITON="1"
NUM_FEWSHOT=""
EXTRA_ARGS=""

usage() {
    cat <<EOF
Usage: bash run_eval.sh --model <path_or_hub_id> [options]

Required:
  --model PATH          Local path or HuggingFace Hub ID of the model

Options:
  --task TASK           lm-eval task name (default: ruler)
  --limit N             Max samples per subtask (default: unlimited)
  --batch-size N        Batch size (default: auto)
  --output DIR          Output directory (default: ./results)
  --keep-ratio F        Fraction of KV pairs to keep (default: 0.1)
  --window-size N       Recent tokens always kept (default: 8)
  --kernel-size N       Score smoothing kernel (default: 11, must be odd)
  --no-triton           Disable Triton acceleration (use PyTorch fallback)
  --num-fewshot N       Number of few-shot examples
  --extra "ARGS"        Additional arguments passed to lm_eval
  -h, --help            Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";       shift 2 ;;
        --task)         TASK="$2";        shift 2 ;;
        --limit)        LIMIT="$2";       shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";  shift 2 ;;
        --output)       OUTPUT="$2";      shift 2 ;;
        --keep-ratio)   KEEP_RATIO="$2";  shift 2 ;;
        --window-size)  WINDOW_SIZE="$2"; shift 2 ;;
        --kernel-size)  KERNEL_SIZE="$2"; shift 2 ;;
        --no-triton)    USE_TRITON="0";   shift ;;
        --num-fewshot)  NUM_FEWSHOT="$2"; shift 2 ;;
        --extra)        EXTRA_ARGS="$2";  shift 2 ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "${MODEL}" ]; then
    echo "Error: --model is required (or set MODEL_PATH in .env)"
    echo ""
    usage
fi

# Resolve model path: if it's a relative/absolute local path, make it absolute
if [ -d "${MODEL}" ]; then
    MODEL="$(cd "${MODEL}" && pwd)"
fi

mkdir -p "${OUTPUT}"

# Build gen_kwargs string (lm_eval 0.4+ uses key=value format)
GEN_KWARGS="cache_implementation=quantized,backend=DropKV"
GEN_KWARGS="${GEN_KWARGS},keep_ratio=${KEEP_RATIO}"
GEN_KWARGS="${GEN_KWARGS},window_size=${WINDOW_SIZE}"
GEN_KWARGS="${GEN_KWARGS},kernel_size=${KERNEL_SIZE}"
GEN_KWARGS="${GEN_KWARGS},use_triton=${USE_TRITON}"

# Build command
CMD=(python -m lm_eval
    --model hf
    --model_args "pretrained=${MODEL},dtype=bfloat16"
    --tasks "${TASK}"
    --batch_size "${BATCH_SIZE}"
    --output_path "${OUTPUT}"
    --gen_kwargs "${GEN_KWARGS}"
)

[ -n "${LIMIT}" ]       && CMD+=(--limit "${LIMIT}")
[ -n "${NUM_FEWSHOT}" ] && CMD+=(--num_fewshot "${NUM_FEWSHOT}")
[ -n "${EXTRA_ARGS}" ]  && CMD+=(${EXTRA_ARGS})

echo "==> Running evaluation"
echo "    Model:      ${MODEL}"
echo "    Task:       ${TASK}"
echo "    Keep ratio: ${KEEP_RATIO}"
echo "    Window:     ${WINDOW_SIZE}"
echo "    Kernel:     ${KERNEL_SIZE}"
echo "    Triton:     $([ "${USE_TRITON}" = "1" ] && echo "enabled" || echo "disabled")"
echo "    Output:     ${OUTPUT}"
[ -n "${LIMIT}" ] && echo "    Limit:      ${LIMIT}"
echo ""
echo "    ${CMD[*]}"
echo ""

"${CMD[@]}"
