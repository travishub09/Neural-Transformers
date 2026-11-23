"""Dense, differentiable gating for the pinned Transformers Qwen3 implementation.

We replace only o_proj and down_proj with weight-sharing Linear subclasses.
Qwen3's own forward methods still perform Q/K normalization, RoPE, GQA, cache
updates, attention, SwiGLU, residuals, and normalization. State-dict keys and
Parameter objects are unchanged. This simulates sparsity; it saves no matmuls.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Iterator, Literal, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from .reordering import CanonicalLinear, ordered_linear

UnitKind = Literal["attention", "mlp"]
UnitKey = tuple[int, UnitKind]
GateMap = Mapping[UnitKey, Tensor]
SUPPORTED_TRANSFORMERS = "4.57.1"


@dataclass(frozen=True)
class UnitActivity:
    """Per-token norms immediately before the gated projection.

    All fields have shape [batch, current_sequence_length, units]. Norms use
    float32 reductions. observed_norms are PRE-gate norms for selected units,
    with zero placeholders for unselected units; contribution_norms are norms
    AFTER multiplication by the gate. Neither is a causal importance label.
    `selected` is a logical selection, not a claim of kernel-level skipping.
    """

    gates: Tensor
    selected: Tensor
    observed_norms: Tensor
    contribution_norms: Tensor


@dataclass
class ActivityTrace:
    # Keep the latest invocation of each layer/component, not an unbounded log.
    records: dict[UnitKey, UnitActivity] = field(default_factory=dict)

    def clear(self) -> None:
        self.records.clear()


@dataclass
class _RunState:
    gates: dict[UnitKey, Tensor] = field(default_factory=dict)
    trace: ActivityTrace | None = None
    detach_observations: bool = True
    active: bool = False


class _GatedProjection(nn.Linear):
    """Keep original weight/bias identity and Linear state-dict names."""

    def __init__(self, original: nn.Linear, units: int, key: UnitKey, state: _RunState):
        # Do not allocate or initialize new weights, or advance the random seed.
        nn.Module.__init__(self)
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias
        self.units = units
        self.unit_width = self.in_features // units
        self.key = key
        self.run_state = state
        self.canonical_mode = getattr(original, 'canonical_mode', None)
        for name in ('canonical_permutation', 'canonical_inverse'):
            value = getattr(original, name, None)
            self.register_buffer(name, None if value is None else value.clone(), persistent=False)
        self.train(original.training)

    def _project(self, x):
        return ordered_linear(x, self.weight, self.bias, self.canonical_mode,
                              self.canonical_permutation, self.canonical_inverse)

    def forward(self, x: Tensor) -> Tensor:
        state = self.run_state
        gate = state.gates.get(self.key)
        if gate is None and state.trace is None:
            return self._project(x)
        if x.ndim != 3:
            raise ValueError("Expected Qwen3 projection input [batch, tokens, channels]")
        grouped = x.reshape(*x.shape[:-1], self.units, self.unit_width)
        if gate is None:
            values = x.new_ones((*x.shape[:-1], self.units))
            gated = grouped
        else:
            if gate.device != x.device:
                raise ValueError(f"Gate {self.key} must be on {x.device}, got {gate.device}")
            if gate.ndim == 3 and any(
                requested not in (1, actual)
                for requested, actual in zip(gate.shape[:2], x.shape[:2])
            ):
                raise ValueError(
                    f"Gate {self.key} shape {tuple(gate.shape)} does not match "
                    f"current batch/token dimensions {tuple(x.shape[:2])}; "
                    "supply gates for the current decode chunk"
                )
            # Casting keeps the autograd connection to FP32 controller outputs.
            values = gate.to(dtype=x.dtype).expand(*x.shape[:-1], self.units)
            gated = grouped * values.unsqueeze(-1)
        if state.trace is not None:
            source = grouped.detach() if state.detach_observations else grouped
            result = gated.detach() if state.detach_observations else gated
            recorded_gates = values.detach() if state.detach_observations else values
            selected = recorded_gates != 0
            observed = torch.linalg.vector_norm(source.float(), dim=-1)
            # No dense counterfactual activity for unselected units enters history.
            observed = torch.where(selected, observed, torch.zeros_like(observed))
            contribution = torch.linalg.vector_norm(result.float(), dim=-1)
            state.trace.records[self.key] = UnitActivity(
                gates=recorded_gates.clone(), selected=selected,
                observed_norms=observed, contribution_norms=contribution,
            )
        return self._project(gated.reshape_as(x))


class GateController:
    """Controls an instrumented model without adding trainable base parameters.

    Use one controller per model and one non-overlapping context at a time.
    Gate tensors are supplied by the caller, not registered as model weights.
    Missing entries mean fully on. Values are booleans or finite floats in [0,1].
    Shapes: [units] (all positions), or [batch, tokens, units]; batch/token axes
    may be 1 for explicit broadcasting. Each cached decode call needs gates
    for its CURRENT chunk. Static [units] gates also work with model.generate().
    """

    def __init__(self, model: Qwen3ForCausalLM, group_size: int):
        if version("transformers") != SUPPORTED_TRANSFORMERS:
            raise RuntimeError(f"This wrapper is validated with transformers=={SUPPORTED_TRANSFORMERS}")
        if not isinstance(model, (Qwen3ForCausalLM, MistralForCausalLM)):
            raise TypeError("Expected non-quantized Qwen3 or Mistral causal LM")
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if model.config.intermediate_size % group_size:
            raise ValueError("group_size must divide intermediate_size")
        if getattr(model, "is_quantized", False):
            raise ValueError("Quantized models are not supported in step 2")
        self.model = model
        self.group_size = group_size
        self.state = _RunState()
        self._originals: list[tuple[nn.Module, str, nn.Linear]] = []
        self._specs: dict[UnitKey, tuple[int, torch.device]] = {}
        self._removed = False
        pending = []
        for idx, layer in enumerate(model.model.layers):
            entries = (
                (layer.self_attn, "o_proj", model.config.num_attention_heads, "attention"),
                (layer.mlp, "down_proj", model.config.intermediate_size // group_size, "mlp"),
            )
            for parent, name, units, kind in entries:
                original = getattr(parent, name)
                if type(original) not in (nn.Linear, CanonicalLinear):
                    raise TypeError("Expected an ungated Linear projection; do not install twice")
                if original.weight.is_meta or original.in_features % units:
                    raise ValueError("Load real, evenly divisible projection weights before installing gates")
                pending.append((parent, name, original, units, (idx, kind)))
        # Validate all layers before making any mutation.
        for parent, name, original, units, key in pending:
            replacement = _GatedProjection(original, units, key, self.state)
            setattr(parent, name, replacement)
            self._originals.append((parent, name, original))
            self._specs[key] = (units, original.weight.device)
        model.requires_grad_(False)

    @property
    def units_per_layer(self) -> dict[str, int]:
        return {kind: self._specs[(0, kind)][0] for kind in ("attention", "mlp")}

    def _validate(self, gates: GateMap) -> dict[UnitKey, Tensor]:
        validated = dict(gates)
        for key, gate in validated.items():
            if key not in self._specs:
                raise ValueError(f"Unknown layer/component key: {key!r}")
            units, device = self._specs[key]
            if not isinstance(gate, Tensor):
                raise TypeError(f"Gate {key} must be a tensor")
            if gate.ndim not in (1, 3) or gate.shape[-1] != units or any(d == 0 for d in gate.shape):
                raise ValueError(f"Gate {key} needs shape [{units}] or [batch, tokens, {units}]")
            if gate.dtype != torch.bool and not gate.is_floating_point():
                raise TypeError("Gates must be floating point or boolean")
            if gate.device != device:
                raise ValueError(f"Gate {key} must be on model device {device}")
            if not bool(torch.isfinite(gate).all()) or not bool(((gate >= 0) & (gate <= 1)).all()):
                raise ValueError("Gate values must be finite and within [0, 1]")
        return validated

    @contextmanager
    def use_gates(
        self, gates: GateMap | None = None, *, record: bool = False,
        detach_observations: bool = True,
    ) -> Iterator[ActivityTrace]:
        """Apply one gate schedule, resetting state even if forward raises.

        Keep backward INSIDE this context when using gradient checkpointing;
        recomputation must see the same gates. Use non-reentrant checkpointing
        with the frozen base model. Do not mutate gate tensors before backward.
        Observations are detached by default; opt into graph-connected history
        explicitly for future controller training. They are not training labels.
        """
        if self._removed:
            raise RuntimeError("This controller was removed")
        if self.state.active:
            raise RuntimeError("Overlapping gate contexts on one model are not supported")
        validated = self._validate({} if gates is None else gates)
        trace = ActivityTrace()
        self.state.gates = validated
        self.state.trace = trace if record else None
        self.state.detach_observations = detach_observations
        self.state.active = True
        try:
            yield trace
        finally:
            self.state.gates = {}
            self.state.trace = None
            self.state.active = False

    def remove(self) -> None:
        """Restore the original projection modules; weights remain frozen."""
        if self.state.active:
            raise RuntimeError("Exit the gate context before removing instrumentation")
        for parent, name, original in self._originals:
            replacement = getattr(parent, name)
            original.train(replacement.training)
            setattr(parent, name, original)
        self._removed = True


def gate_qwen3(model: Qwen3ForCausalLM, group_size: int = 128) -> GateController:
    """Instrument and freeze an already loaded Qwen3 model in place."""
    return GateController(model, group_size)


def gate_mistral(model: MistralForCausalLM, group_size: int = 1) -> GateController:
    """Preserve Mistral's native attention/cache; gate individual SwiGLU outputs.

    Dense masking is algebraically equivalent to omitting a neuron's gate/up
    rows and down column under hard masks; actual kernels still run densely.
    """
    if not isinstance(model, MistralForCausalLM):
        raise TypeError('Expected MistralForCausalLM')
    return GateController(model, group_size)
