# Neural-Transformers

Code and results for **"Neural-Transformers: Predicting a Token's Full Neuron Path Ahead of Time in SwiGLU Language Models"** (Travis Waters, 2026). The paper is in [`paper/`](paper/planning_the_path.pdf).

The **upfront path planner** is a small causal transformer whose positions are the layers of a frozen SwiGLU model. After two dense layers it generates a per-neuron MLP mask for every remaining layer of the current token, one layer at a time, so the weights a token needs are known before those layers run. It is trained end to end on output fidelity (cosine + KL distillation with a compute penalty) rather than by imitating activation labels.

| Mistral-7B, ~50% of MLP neurons kept | Perplexity increase | HellaSwag (dense 81.18) |
|---|---:|---:|
| Upfront planner | **+5.21%** | **79.49** |
| GRIFFIN | +72.47% | 74.10 |
| NeuroPrefetcher (authors' released predictor) | +227.96% | 60.92 |

All quality results simulate skipping by masking a dense computation. The paper's deployment checks (Section 7) report the open problems: retention calibrated on general text does not transfer to GSM8K/MMLU, and a straightforward PyTorch implementation cuts CPU-to-GPU weight traffic by 52–65% but decodes more slowly than dense.

## Repository layout

| Path | Contents |
|---|---|
| `pathpred/` | Library: gated SwiGLU models, the planner, training objectives, calibration, baselines |
| `scripts/` | Experiment entry points (latest code; produced the Section 7 follow-ups and the second Qwen seed) |
| `snapshots/` | Verbatim code from the commit each earlier experiment actually ran from; see [`snapshots/README.md`](snapshots/README.md) |
| `configs/` | Every experiment configuration, with model IDs and revision hashes |
| `data/` | Corpus manifests and the 63 test-document hashes |
| `results/` | The result files the paper cites (summaries, per-condition JSON) |
| `docs/paper-numbers.md` | Every printed number → source file and JSON pointer, protocol and realized retention |
| `tests/` | Unit tests for the library |
| `environment/` | Full package freeze of the machines used |

## Setup

```bash
pip install -r requirements.txt
pytest -q                 # unit tests; no GPU needed for most
```

Models: `mistralai/Mistral-7B-v0.1` (revision `27d67f1b5f57dc0953326b2601d68371d40ea8da`) and `Qwen/Qwen3-1.7B-Base` (revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`). Both stay frozen. Hardware: one A40 for Qwen experiments, one A100 80GB for Mistral training and the deployment checks.

## Data

Training uses FineWeb-Edu, codeparrot-clean and UltraChat-200k in an 80/10/10 token ratio, split by document with a fixed salt (`configs/*-corpus.json`). `scripts/prepare_corpus.py` (Qwen) and `scripts/prepare_mistral_data.py` (Mistral) rebuild the splits deterministically; `data/corpus-manifest.json` and `data/full-manifest.json` record the resulting token counts and file hashes so a rebuild can be checked. `data/test-documents.json` lists the SHA-256 of each of the 63 test documents. HellaSwag, GSM8K, MMLU and WikiText-2 come from their standard Hugging Face releases (`configs/`).

## Reproducing the paper

Commands are the stages used in our runs. They expect a prepared corpus and the base model under `data/` and `checkpoints/`. For results produced before the latest code, run the command from the matching snapshot folder for an exact match.

| Paper result | Code | Stages |
|---|---|---|
| Mistral planner training (Fig. 2) | `snapshots/mistral` | `python -m scripts.mistral_full copy`, `… scheduled`, `… e2e --budget 50`; then `python -m scripts.mistral_phased e2e --budget 50`, `… --budget 40`; `python -m scripts.mistral_lower e2e --budget 30`, `… --budget 20`; each followed by `evaluate` |
| Table 1 (continuation, GRIFFIN, NeuroPrefetcher) | `snapshots/mistral` | `python -m scripts.mistral_comparisons calibrate\|evaluate --kind {planner,griffin,neuro} --budget {50,40} --prompt 64`; long prompts: `python -m scripts.mistral_long_prompt run` |
| Table 2 (HellaSwag) | `snapshots/mistral` | `python -m scripts.mistral_hellaswag_matched run` and `… run --lower` |
| Table 3 (imitation vs end to end, Qwen) | `snapshots/qwen-e2e-matched` | `python -m scripts.neuron_guessing train\|evaluate`; `python -m scripts.e2e_full --signal {binary,hidden}` |
| Table 4 (layer by layer vs one-shot), seed 1 | `snapshots/qwen-planner` | `python -m scripts.planner_experiment --variant {autoregressive,one_shot_hidden,one_shot_history} --stage {copy,scheduled,e2e} --budget {50,40}` |
| Table 4, seed 2 | top level | `python -m scripts.followup_seed --config configs/followup-seed-v3.json` |
| Table 5 (sequence prior) | `snapshots/sequence-prior` | `python -m scripts.sequence_prior_experiment --variant {baseline,control,warm_prior,prior_first} --budget {50,40}` |
| Section 6.3 granularity | top level | `python -m scripts.calibrate_sparsity --stage {calibration,grid}`; `python -m scripts.quick_fresh_groups` |
| Section 6.6 coarser gates | `snapshots/hierarchy-gates` | `python -m scripts.hierarchy_experiment --config configs/hierarchy-pilot.json --variant {A,B,C,D} --budget 50` |
| Deterministic-gate ablation (Sec. 3.3) | `snapshots/hard-gates` | `python -m scripts.hard_gate_comparison prepare\|train\|report` |
| NeuroPrefetcher on Qwen, nearest-pattern control | `snapshots/neuroprefetcher` | `python -m scripts.neuro_data`, then `python -m scripts.neuro_evaluate --budget {0.5,0.4}` |
| Table 6 (GSM8K, MMLU) | top level | `python -m scripts.followup_reasoning --config configs/followup-reasoning-v2.json` |
| Table 7 (latency, weight traffic) | top level | `python -m scripts.followup_latency --config configs/followup-latency-v3.json` |
| Self-speculative decoding | top level | `python -m scripts.followup_speculative --config configs/followup-speculative.json` |
| Context-length check (Sec. 5.1) | top level | `python -m scripts.followup_positions --config configs/followup-positions.json` |

This clean repository has not yet been re-run end to end; please open an issue if a stage does not run.

## Checking every number in the paper

```bash
REPO_ROOT=. python3 scripts/verify_paper_numbers.py docs/paper-numbers-spec.py
```

This opens each cited result file, resolves the JSON pointer or formula for every printed number (330 rows), and fails if any value does not round to what the paper prints.

## Trained planners

Planner checkpoints (Mistral 30%/20%/40%/50%, Qwen 50%/40%) will be published on Hugging Face; the link will be added here.

## Citation

```bibtex
@misc{waters2026transformers,
  title  = {Neural-Transformers: Predicting a Token's Full Neuron Path Ahead of Time in SwiGLU Language Models},
  author = {Waters, Travis},
  year   = {2026},
  note   = {Preprint}
}
```

Code is released under the MIT License. The paper was drafted with AI assistance; see its acknowledgments.
