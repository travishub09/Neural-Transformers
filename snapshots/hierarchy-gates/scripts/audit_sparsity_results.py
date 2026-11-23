"""Independent CPU reconstruction of every condition's saved quality metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('artifacts/sparsity-calibration-v2'))
    a=p.parse_args();total=0;windows=0
    for stage in ['calibration','grid']:
        r=json.loads((a.root/f'{stage}-results.json').read_text())
        manifest=json.loads((a.root/f'{stage}-manifest.json').read_text())
        docs=[json.loads(f.read_text()) for f in a.root.glob(f'{stage}-window-*.json')]
        assert len(docs)==r['documents']==63
        assert len({d['metadata']['document_sha256'] for d in docs})==63
        assert all(d['fingerprint']==r['fingerprint'] for d in docs)
        for source,digest in r['fingerprint'].items():
            if Path(source).exists():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
        for doc in docs:
            assert set(doc['conditions'])==set(r['results'])
            dense={row['teacher_nll'] for row in doc['conditions'].values()}
            assert len(dense)==1
        for name,result in r['results'].items():
            rows=[d['conditions'][name] for d in docs]
            for domain in ['overall','web','code','conversation']:
                group=rows if domain=='overall' else [row for row in rows if row['domain']==domain]
                expected=result['overall'] if domain=='overall' else result['domains'][domain]
                n=sum(row['tokens'] for row in group)
                t=sum(row['teacher_nll'] for row in group)/n
                s=sum(row['student_nll'] for row in group)/n
                actual={'dense_nll':t,'student_nll':s,'dense_perplexity':math.exp(t),
                        'student_perplexity':math.exp(s),'relative_ppl_increase':math.expm1(s-t),
                        'teacher_kl_nats':sum(row['kl'] for row in group)/n,
                        'top1_agreement':sum(row['top1_agreement'] for row in group)/n}
                for key,value in actual.items():
                    assert math.isfinite(value)
                    assert math.isclose(expected[key],value,rel_tol=1e-10,abs_tol=1e-10),(name,key)
            c=result['condition']
            for kind,units in [('mlp',6144//c['group_size']),('attention',16)]:
                fraction=math.ceil(units*c['keep'])/units if c['component'] in [kind,'both'] else 1
                if c['protection'] in ['edges','both']:fraction=(4+24*fraction)/28
                assert math.isclose(result['retained'][kind],fraction,abs_tol=1e-10),(name,kind)
                assert all(math.isclose(row['retained'][kind],fraction,abs_tol=1e-10) for row in rows)
            total+=1
        assert manifest['status']=='complete'
        windows+=len(docs)
    cal=json.loads((a.root/'calibration-statistics.json').read_text())
    calibration_file=a.root/cal['file']['file']
    assert hashlib.sha256(calibration_file.read_bytes()).hexdigest()==cal['file']['sha256']
    train={d['document_sha256'] for d in cal['documents']}
    test={d['metadata']['document_sha256'] for d in docs}
    assert not train&test and all(d['split']=='train' for d in cal['documents'])
    audit=json.loads((a.root/'harness-audit.json').read_text())
    assert audit['dense_logits_bitwise_unchanged_after_precision_audit']
    for precision in ['bf16','fp32']:
        assert all(v['wrapper_direct_bitwise_equal'] and v['cached_wrapper_direct_bitwise_equal']
                   for v in audit[precision].values())
    output={'passed':True,'conditions_reconstructed':total,'stage_document_records':windows,
            'all_retention_counts_verified':True,'source_fingerprints_verified_where_local':True,
            'calibration_documents_disjoint_from_test':True,'harness_audit_verified':True}
    output['calibration_tensor_checksum_verified']=True
    (a.root/'independent-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':main()
