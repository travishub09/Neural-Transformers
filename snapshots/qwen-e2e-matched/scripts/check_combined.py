"""Local architecture accounting and exact saved-checkpoint warm-start check."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, file_sha256
from pathpred.combined_prediction import CombinedNeuronPredictor, make_predictor
from pathpred.neuron_prediction import NeuronPredictor, neuron_config


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261112)
    result = {'accounting': {}, 'warm_starts': {}}
    for signal in ['binary', 'hidden', 'combined']:
        with torch.device('meta'):
            p = make_predictor(signal)
        info = p.accounting()
        info['bf16_weight_payload_bytes'] = info['parameters'] * 2
        info['weight_payload_fraction_of_qwen'] = info['parameters'] / 1720574976
        info['matmul_fraction_of_dense_1k_reference'] = info['matmul_flops_per_llm_token'] / 3675783168
        result['accounting'][signal] = info
        del p
    for budget in [50, 40]:
        checkpoint = Path(f'artifacts/e2e-full-v1/budget-{budget}/predictor-10000000.safetensors')
        meta = json.loads(checkpoint.with_name('checkpoint-10000000.json').read_text())
        digest = file_sha256(checkpoint)
        assert digest == meta['file']['sha256']
        path = NeuronPredictor(neuron_config()).eval()
        path.load_state_dict(load_file(str(checkpoint)))
        p = CombinedNeuronPredictor().eval()
        p.load_path_state(path.state_dict())
        states = torch.randint(-1, 2, (1, 2, 28, 6144), dtype=torch.int8)
        hidden = torch.randn(1, 2, 28, 2048)
        with torch.no_grad():
            expected = path.forward_tokens(states=states)
            actual = p.forward_tokens(states=states, hidden=hidden)
        assert torch.equal(expected, actual)
        result['warm_starts'][str(budget)] = {
            'checkpoint_sha256': digest, 'exact_logits_equal': True,
            'maximum_logit_difference': float((actual - expected).abs().max()),
            'scope': 'Actual trained predictor weights; synthetic observations on CPU, not a Qwen quality evaluation'}
        del p, path
    atomic_json(Path('artifacts/combined-v1/local-preflight.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
