"""Capacity-constrained Lloyd clustering, reversible MLP permutations, and costs.

Neurons are observations; sampled token activations are their features.
Each assignment is an exact min-cost flow for integer-rounded distances.
The overall k-means problem is nonconvex: restarts do not prove a global optimum.
"""
import numpy as np
import torch
from ortools.graph.python import min_cost_flow


def activation_features(raw):
    """[positions, neurons] -> [neurons, positions], magnitude co-activity.

    Scaling is per neuron, with no centering or fitted cross-split statistics.
    A zero-activity neuron is represented by an all-zero feature vector.
    """
    if raw.ndim != 2 or min(raw.shape) == 0 or not torch.isfinite(raw).all():
        raise ValueError("Expected nonempty finite [positions, neurons]")
    x = raw.float().abs().T.contiguous()
    return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)


def balanced_assignment(distances, group_size, scale=1_000_000):
    """Return one cluster per neuron, exactly group_size neurons per cluster."""
    d = np.asarray(distances, dtype=np.float64)
    if (d.ndim != 2 or group_size < 1 or scale < 1 or
            d.shape[0] != d.shape[1] * group_size or not np.isfinite(d).all()):
        raise ValueError("Invalid balanced assignment dimensions/costs")
    n, k = d.shape
    # Row offsets do not affect the solution and avoid negative costs.
    shifted = d - d.min(axis=1, keepdims=True)
    if shifted.max(initial=0) * scale > 1e12:
        raise ValueError("Integer costs would be too large")
    costs = np.rint(shifted * scale).astype(np.int64)
    flow = min_cost_flow.SimpleMinCostFlow()
    arcs = flow.add_arcs_with_capacity_and_unit_cost(
        np.repeat(np.arange(n), k), n + np.tile(np.arange(k), n),
        np.ones(n * k, dtype=np.int64), costs.ravel())
    flow.set_nodes_supplies(np.arange(n + k),
                           np.r_[np.ones(n, dtype=np.int64), np.full(k, -group_size)])
    if flow.solve() != flow.OPTIMAL:
        raise RuntimeError("Balanced min-cost assignment failed")
    selected = flow.flows(arcs).reshape(n, k)
    if not (np.all(selected.sum(1) == 1) and np.all(selected.sum(0) == group_size)):
        raise RuntimeError("Invalid solver flow")
    return torch.from_numpy(selected.argmax(1)).long()


def squared_distances(x, centers):
    return (x.square().sum(1, keepdim=True) + centers.square().sum(1)[None]
            - 2 * (x @ centers.T)).clamp_min(0)


def balanced_kmeans(x, group_size, *, seed, restarts=2, max_iterations=30,
                    relative_tolerance=1e-5, distance_integer_scale=1_000_000):
    if (x.ndim != 2 or group_size < 1 or x.shape[0] % group_size or min(x.shape) == 0 or
            restarts < 1 or max_iterations < 1 or not torch.isfinite(x).all()):
        raise ValueError("Invalid features or clustering settings")
    n, _ = x.shape
    k = n // group_size
    best = None
    runs = []
    for restart in range(restarts):
        rng = torch.Generator(device=x.device).manual_seed(seed + restart)
        # k-means++ initialization. No held-out samples enter any choice.
        first = int(torch.randint(n, (1,), generator=rng, device=x.device))
        seeds = [first]
        nearest = squared_distances(x, x[first:first + 1]).squeeze(1)
        for _ in range(1, k):
            nearest[seeds] = 0
            if float(nearest.sum()) > 0:
                nxt = int(torch.multinomial(nearest, 1, generator=rng))
            else:
                nxt = next(i for i in range(n) if i not in seeds)
            seeds.append(nxt)
            nearest = torch.minimum(nearest, squared_distances(x, x[nxt:nxt + 1]).squeeze(1))
        centers = x[seeds].clone()
        previous = None
        history = []
        run_best = None
        stop_reason = "iteration_limit"
        for iteration in range(max_iterations):
            distances = squared_distances(x, centers)
            labels = balanced_assignment(distances.cpu().numpy(), group_size,
                                         distance_integer_scale).to(x.device)
            # Avoid nondeterministic atomic index_add reductions on CUDA.
            centers = torch.stack([x[labels == j].mean(0) for j in range(k)])
            objective = float((x - centers[labels]).square().sum())
            history.append(objective)
            if run_best is None or objective < run_best[0]:
                run_best = (objective, labels.clone())
            if previous is not None and torch.equal(labels, previous):
                stop_reason = "unchanged_assignment"
                break
            if len(history) > 1 and abs(history[-2] - objective) <= relative_tolerance * max(history[-2], 1e-12):
                stop_reason = "relative_objective_tolerance"
                break
            previous = labels.clone()
        runs.append({"restart": restart, "objectives": history, "stop_reason": stop_reason,
                     "best_objective": run_best[0]})
        if best is None or run_best[0] < best[0]:
            best = (*run_best, restart)
    labels = best[1].cpu()
    permutation = torch.argsort(labels, stable=True)
    return permutation, {"objective": best[0], "selected_restart": best[2], "runs": runs}


def validate_permutation(permutation, size):
    if (permutation.dtype != torch.int64 or permutation.ndim != 1 or
            permutation.numel() != size or
            not torch.equal(permutation.cpu().sort().values, torch.arange(size))):
        raise ValueError("Expected a bijection over neuron indices")


@torch.no_grad()
def reorder_mlp_(mlp, permutation):
    """new intermediate[j] = old intermediate[permutation[j]].

    Permute gate/up rows and biases, down columns. Preserve Parameter objects.
    Apply argsort(permutation) to restore the original weights exactly.
    """
    width = mlp.gate_proj.out_features
    validate_permutation(permutation, width)
    if mlp.up_proj.out_features != width or mlp.down_proj.in_features != width:
        raise ValueError("Inconsistent SwiGLU dimensions")
    p = permutation.to(mlp.gate_proj.weight.device)
    for proj in (mlp.gate_proj, mlp.up_proj):
        proj.weight.copy_(proj.weight.index_select(0, p))
        if proj.bias is not None:
            proj.bias.copy_(proj.bias.index_select(0, p))
    mlp.down_proj.weight.copy_(mlp.down_proj.weight.index_select(1, p))


def grouping_metrics(features, permutation, group_size):
    """Descriptive within-group cosine and SSE; no outcome-based fitting."""
    validate_permutation(permutation, features.shape[0])
    if group_size < 2 or features.shape[0] % group_size:
        raise ValueError("Invalid group size")
    grouped = features[permutation.to(features.device)].reshape(-1, group_size, features.shape[1])
    centers = grouped.mean(1)
    sse = (grouped - centers[:, None]).square().sum()
    # Sum all off-diagonal dot products without an N x N similarity matrix.
    cos = (grouped.sum(1).square().sum(1) - grouped.square().sum((1, 2))) / (group_size * (group_size - 1))
    return {"mean_pairwise_magnitude_cosine": float(cos.mean()), "sse": float(sse),
            "zero_activity_neurons": int((features.square().sum(1) == 0).sum())}


def unit_costs(config, group_size=128, context_lengths=(1024, 16384), weight_bytes=2):
    """Ideal batch-one decoding matmul costs, not measured hardware savings.

    Query gates can skip Q/O and per-query attention; K/V remain mandatory.
    MLP gates can theoretically skip gate/up/down slices with future kernels.
    One multiply-add counts as two FLOPs. Includes separate mandatory LM head.
    """
    h, d, q, kv = config.hidden_size, config.head_dim, config.num_attention_heads, config.num_key_value_heads
    layers, m = config.num_hidden_layers, config.intermediate_size
    if m % group_size or q % kv or weight_bytes < 1 or any(c < 1 for c in context_lengths):
        raise ValueError("Invalid cost configuration")
    groups = m // group_size
    params = [2 * h * d] * q + [3 * h * group_size] * groups
    fixed_kv_params = 2 * h * kv * d
    lm_params = h * config.vocab_size
    scenarios = {}
    for context in context_lengths:
        flops = [4 * h * d + 4 * context * d] * q + [6 * h * group_size] * groups
        gated = layers * sum(flops)
        mandatory = 2 * layers * fixed_kv_params + 2 * lm_params
        scenarios[str(context)] = {
            "per_layer_unit_flops": flops,
            "all_layers_gated_flops": gated,
            "mandatory_kv_projection_flops": 2 * layers * fixed_kv_params,
            "mandatory_lm_head_flops": 2 * lm_params,
            "dense_accounted_flops": gated + mandatory,
            "mandatory_fraction_of_accounted_flops": mandatory / (gated + mandatory),
        }
    return {
        "layers": layers, "query_heads": q, "kv_heads": kv, "mlp_groups": groups,
        "group_size": group_size, "unit_order": "query heads then MLP groups, repeated per layer",
        "per_layer_unit_weight_bytes": [weight_bytes * n for n in params],
        "mandatory_kv_weight_bytes_all_layers": weight_bytes * layers * fixed_kv_params,
        "mandatory_lm_head_weight_bytes": weight_bytes * lm_params,
        "embedding_lookup_bytes_per_token": weight_bytes * h,
        "query_head_to_kv_head": [i // (q // kv) for i in range(q)],
        "scenarios": scenarios,
        "assumptions": ["Batch-one decode; context includes the current token; multiply-add = 2 FLOPs",
                        "All K/V projections and full KV cache are retained even when query gates are off",
                        "Weight bytes are ideal payload reads, not measured DRAM traffic; caching is ignored",
                        "LM head is always computed; tied embedding/head weights are not double-counted as storage",
                        "Excludes norms, RoPE, softmax, SiLU, elementwise ops, biases, router and kernel overhead",
                        "Cost budget fraction for gated units is not total model FLOPs fraction",
                        "Current wrappers are dense gate simulation and skip no matmuls"],
    }
