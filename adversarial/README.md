# Adversarial perturbation CLI

`vlm-eps-cli.py` learns a tiny, human-imperceptible mask that forces Qwen3-VL to change its answer for a single image via backpropagation.

## Setup
- Python 3.9+ with `torch` and `transformers>=4.57.0` installed (match the repo’s venv guidance).
- Model defaults to `Qwen/Qwen3-VL-2B-Instruct` and auto-selects `cuda` → `mps` → `cpu` unless `--device` is set.

## Usage
```bash
python adversarial/vlm-eps-cli.py \
  --image /path/to/image.jpg \
  --target-text "dog" \
  --prompt "Answer with a single-word label for the animal in this picture." \
  --steps 60 \
  --epsilon 0.0157 \
  --save /tmp/adv.png
```

Key flags:
- `--target-text`: text you want the model to output after the perturbation.
- `--epsilon`: L_inf cap in raw pixel space (default 4/255). Increase slowly if the attack stalls.
- `--device`: `auto|cuda|mps|cpu`; use `auto` to pick the fastest available backend.
- `--save`: optional path to write the reconstructed adversarial image.
- `--max-new-tokens`: trim generation length for faster loops on small prompts.

## What to expect
- The script prints the baseline answer, loss/log-prob trajectory, and the final answer after optimization.
- Max per-pixel delta is reported to keep changes visually minimal.
- Keep images modest in size for faster runs; the attack operates on the model’s patchized `pixel_values`.
