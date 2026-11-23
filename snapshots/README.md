# Code snapshots, exactly as run

The top-level `pathpred/` and `scripts/` are the latest code, which produced the Mistral follow-up experiments (Section 7, second Qwen seed). Earlier experiments ran from research branches whose shared modules diverged. Each folder below is a verbatim copy of `pathpred/`, the experiment scripts and `configs/` from the internal commit recorded in that experiment's results. Run a snapshot's scripts from inside its folder (for example `cd snapshots/sequence-prior && python -m scripts.<name>`). Cloud-pod, billing and monitoring scripts are omitted.

| Snapshot | Internal commit | Result collections | Paper content | Files |
|---|---|---|---|---|
| `hierarchy-gates` | `58d122b2a841` | hierarchy--hierarchy-v1 | Section 6.6 coarser gates (neuron, layer and group gates) | 85 |
| `hard-gates` | `5d84379fa94d` | hard-gates--hard-gate-comparison-v1 | Section 3.3 deterministic-gate ablation | 90 |
| `sequence-prior` | `253353a1001b` | sequence-prior--sequence-prior-v1 | Section 6.5 and Table 5 | 93 |
| `neuroprefetcher` | `76335aa6bb02` | neuro--neuroprefetcher-v1, neuro--neuro-quality-v1, neuro--neuro-controls-v1 | Qwen NeuroPrefetcher reimplementation, nearest-pattern control, relaxed Qwen comparison, Mistral reference reproduction | 107 |
| `qwen-planner` | `46fccb793b28` | planner--planner-v1 | Qwen planner and one-shot controls, Table 4 first seed | 87 |
| `qwen-e2e-matched` | `ec2bd53eda85` | core--e2e-matched-v1 | Table 3 imitation vs end-to-end | 79 |
| `mistral` | `c46a0cac619b` | mistral--mistral-phased-v1, mistral--mistral-lower-v1, mistral--mistral-comparisons-v1 | Mistral planner training and Tables 1-2, Figure 2 | 118 |

Some snapshot configs and scripts contain absolute `/workspace/...` paths from the cloud machines they ran on (for example the location of the NeuroPrefetcher reference checkout). Adjust these to your own paths.
