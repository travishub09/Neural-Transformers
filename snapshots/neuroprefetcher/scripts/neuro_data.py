"""Read-only matching to planner data; never invoke its trainer or evaluator."""
import hashlib
import json
import math
from pathlib import Path
from scripts.neuron_guessing import data


def ordered(records, heldout, c):
    forbidden = {r['document_sha256'] for r in heldout if r['split'] != 'train'}
    rows = [i for i, r in enumerate(records) if r['split'] == 'train' and
            r['length'] >= c['sequence_length'] and r['document_sha256'] not in forbidden]
    rows.sort(key=lambda i: hashlib.sha256((str(c['seed'])+':'+records[i]['document_sha256']+':'+
                                          str(records[i]['document_token_offset'])).encode()).hexdigest())
    return rows[:8], rows[8:]


def stage_windows(order, tokens, source, stage):
    result = []; seen = 0
    for index in order:
        length = min(512, tokens-seen)
        if length <= 0: break
        result.append({'source': source, 'index': index, 'length': length, 'stage': stage})
        seen += length
    assert seen == tokens
    return result


def load_matching(config):
    pc = json.loads(Path(config['planner_config']).read_text())
    prior_root=Path(config['planner_artifact_root'])
    prior_protocol=json.loads((prior_root/'launch/protocol.json').read_text())
    assert prior_protocol['config']==pc, 'Planner configuration differs from the saved run'
    _, heldout = data('data/stage3-corpus-v1')
    sources = {}; audits = {}; orders = {}
    for name in ('copy', 'quality'):
        source = pc[name+'_corpus']; corpus, records = data(source)
        calibration, order = ordered(records, heldout, pc)
        sources[name] = (corpus['input_ids'], records); orders[name] = order
        audits[name] = {'source': source, 'reserved_calibration_windows': calibration,
            'order_sha256': hashlib.sha256(json.dumps(order).encode()).hexdigest(),
            'available_full_windows': len(order)}
        prior=json.loads((prior_root/(name+'-data-audit.json')).read_text())
        for field,value in audits[name].items():
            assert prior[field]==value, f'Saved planner {name} data mismatch: {field}'
    windows = stage_windows(orders['copy'], pc['copy_tokens'], 'copy', 'copy')
    windows += stage_windows(orders['copy'][math.ceil(pc['copy_tokens']/512):],
                              pc['scheduled_tokens'], 'copy', 'scheduled')
    windows += stage_windows(orders['quality'], pc['e2e_tokens_per_budget'], 'quality', 'quality')
    assert sum(row['length'] for row in windows) == config['training_tokens_per_predictor']
    audits['windows_sha256'] = hashlib.sha256(json.dumps(windows, sort_keys=True).encode()).hexdigest()
    audits['training_token_presentations'] = sum(row['length'] for row in windows)
    audits['planner_files_read_only'] = True
    return sources, windows, audits


def window_ids(sources, row, device='cuda'):
    tokens, records = sources[row['source']]; record = records[row['index']]
    return tokens[record['sequence_id']:record['sequence_id']+1,:row['length']].long().to(device)
