R='results/'
MC=R+'mistral--mistral-comparisons-v1/raw/'
P2_64=MC+'prefix-2-prompt-64/budget-%s/documents/%s.json'
LP=MC+'long-prompt-v1/prompt-%s/%s-%s.json'
HS=MC+'hellaswag/%s.json'; HSM=MC+'hellaswag-matched-v1/evaluation/planner-%s.json'; HSL=MC+'hellaswag-lower-v1/results.json'
PH=R+'mistral--mistral-phased-v1/raw/budget-%s/%s.json'; LO=R+'mistral--mistral-lower-v1/raw/%s/%s.json'
QP=R+'planner--planner-v1/raw/%s/budget-%s/%s.json'; Q2=R+'qwen--planner-seed2-v3/raw/%s/budget-%s/evaluation.json'
SP=R+'sequence-prior--sequence-prior-v1/raw/results.json'
EM='docs/e2e-matched/raw/'; NG=R+'core--neuron-guessing-v1/raw/evaluation.json'
SC2=R+'core--sparsity-calibration-v2/raw/%s.json'; FG=R+'core--fresh-groups-quick-v1/raw/'
NQ=R+'neuro--neuro-quality-v1/raw/results.json'; NO=R+'neuro--neuro-controls-v1/raw/oracle/evaluation.json'
RQ=R+'neuro--neuro-controls-v1/raw/relaxed-qwen/results.json'; NM=R+'neuro--neuro-mistral-v1/raw/results.json'
PZ=R+'core--phase0-strong-v1/raw/evaluation-progress.json'; HI=R+'hierarchy--hierarchy-v1/raw/%s/budget-50/%s.json'
HG=R+'hard-gates--hard-gate-comparison-v1/raw/results.json'; NPC=R+'neuro--neuroprefetcher-v1/raw/comparison.json'
RS=R+'mistral--reasoning-v2/raw/summary.json'; LT=R+'mistral--gather-latency-v3/raw/summary.json'; LTD=R+'mistral--gather-latency-v3/raw/%s-documents.json'
SS=R+'mistral--self-speculative-v1/raw/summary.json'; PL=R+'mistral--position-losses-v1/raw/%s.json'
O='/overall/'
CONT='Mistral continuation, 64-token dense prompt, 63 test docs'; WT='Mistral continuation, WikiText-2 64 windows, %s-token prompt, final 512 scored'
WD='Mistral whole-document, 63 test docs'; QWD='Qwen whole-document, 63 test docs'; HSP='Mistral HellaSwag 10,042 val, zero-shot, answer tokens sparse'
DOC='document bootstrap, single seed'
ROWS=[]
def row(p,src,kind='pct',proto='',ret='',note=DOC): ROWS.append((p,src,kind,proto,ret,note))
def d(expr,*refs): return ('derive',expr,list(refs))
def c(desc): return ('const',desc)
# ---- Table 1 / 5.1
for b,pl,gr,np in (('50','5.21','72.47','227.96'),('40','7.29','117.10','594.59')):
    for m,val in (('planner',pl),('griffin',gr),('neuro',np)):
        f=P2_64%(b,m); row(val,(f,O+'relative_ppl_increase'),'pct',CONT,(f,O+'routed_kept'))
row('49.15',(P2_64%('50','planner'),O+'routed_kept'),'pct',CONT,'(retention)')
row('41.27',(P2_64%('40','planner'),O+'routed_kept'),'pct',CONT,'(retention)')
for pr,vals in (('512',(('planner','50','10.62'),('planner','40','14.52'),('griffin','50','71.44'),('griffin','40','114.00'),('neuro','50','169.76'),('neuro','40','490.90'))),
                ('1024',(('planner','50','10.50'),('planner','40','14.10'),('griffin','50','66.62'),('griffin','40','106.48'),('neuro','50','157.45'),('neuro','40','475.13')))):
    for m,b,v in vals:
        f=LP%(pr,m,b); row(v,(f,O+'relative_ppl_increase'),'pct',WT%pr,(f,O+'routed_kept'))
for pr,b,v in (('512','50','48.4'),('1024','50','48.2'),('512','40','40.3')):
    row(v,(LP%(pr,'planner',b),O+'routed_kept'),'pct',WT%pr,'(retention, printed as 48.2–48.4% / 40.3%)')
row('71.4',(LP%('512','griffin','50'),O+'relative_ppl_increase'),'pct',WT%'512',(LP%('512','griffin','50'),O+'routed_kept'))
row('66.6',(LP%('1024','griffin','50'),O+'relative_ppl_increase'),'pct',WT%'1024',(LP%('1024','griffin','50'),O+'routed_kept'))
row('10.6',(LP%('512','planner','50'),O+'relative_ppl_increase'),'pct',WT%'512',(LP%('512','planner','50'),O+'routed_kept'))
row('10.5',(LP%('1024','planner','50'),O+'relative_ppl_increase'),'pct',WT%'1024',(LP%('1024','planner','50'),O+'routed_kept'))
row('157.4',(LP%('1024','neuro','50'),O+'relative_ppl_increase'),'pct',WT%'1024',(LP%('1024','neuro','50'),O+'routed_kept'))
f=MC+'prefix-2-prompt-0/budget-50/documents/%s.json'
row('277.6',(f%'neuro',O+'relative_ppl_increase'),'pct','Mistral whole-document (no prompt), 2 dense layers, 63 test docs',(f%'neuro',O+'routed_kept'))
row('48.06',(f%'neuro',O+'routed_kept'),'pct','same','(retention)')
row('5.41',(f%'planner',O+'relative_ppl_increase'),'pct','Mistral whole-document (no prompt), 2 dense layers, 63 test docs',(f%'planner',O+'routed_kept'))
row('49.49',(f%'planner',O+'routed_kept'),'pct','same','(retention)')
f=MC+'prefix-20-prompt-0/budget-50/documents/%s.json'
for m,v,rv in (('planner','5.75','48.70'),('neuro','38.61','49.66')):
    row(v,(f%m,O+'relative_ppl_increase'),'pct','Mistral whole-document, NeuroPrefetcher config (20 dense layers, dense down proj)',(f%m,O+'routed_kept'))
    row(rv,(f%m,O+'routed_kept'),'pct','same','(retention)')
row('10.29',(PL%'summary','/planner-50-prompt-64-cal-512/relative_ppl_increase'),'pct',WT%'64'+', fixed prompt-512 offsets',(PL%'summary','/planner-50-prompt-64-cal-512/routed_kept'))
row('10.89',(PL%'summary','/planner-50-prompt-1024-cal-512/relative_ppl_increase'),'pct',WT%'1024'+', fixed prompt-512 offsets',(PL%'summary','/planner-50-prompt-1024-cal-512/routed_kept'))
for v,ptr in (('0.005','/50-1024-minus-64/excess_nll_difference/mean'),('0.001','/50-1024-minus-64/excess_nll_difference/ci95/0'),('0.010','/50-1024-minus-64/excess_nll_difference/ci95/1')):
    row(v,(PL%'paired',ptr),'raw','paired excess NLL (nats), prompt 1024 minus 64, 50% planner','see summary rows','paired non-overlapping block bootstrap, single seed')
# retention drift (derived from per-position file)
pos=PL%'planner-50-prompt-64-cal-512-positions'
row('57',d("100*sum(sum(doc['metrics'][i][5] for i in range(1,33)) for doc in v[0]['documents'])/(32*len(v[0]['documents']))",(pos,'')),'raw',WT%'64','(retention, first 32 continuation tokens)','derived mean over 64 windows')
row('48',d("100*sum(sum(doc['metrics'][i][5] for i in range(len(doc['metrics'])-32,len(doc['metrics']))) for doc in v[0]['documents'])/(32*len(v[0]['documents']))",(pos,'')),'raw',WT%'64','(retention, last 32 continuation tokens)','derived mean over 64 windows')
row('42.5',d('100*(v[0]/v[1]-1)',(NM,'/20/paper_denominator_ppl'),(NM,'/32/paper_denominator_ppl')),'raw',"authors' 2,048-token WikiText slice, released engine, 20 dense layers",(NM,'/20/suffix_kept'),'single evaluation slice')
row('49.63',(NM,'/20/suffix_kept'),'pct','same','(retention)')
row('3.9934',(NM,'/32/paper_denominator_ppl'),'raw','same, 32 dense layers','100%','reproduction of saved reference 3.9937 (wikitext-reference.json /reported_ppl/32)')
row('3.9937',(MC+'wikitext-reference.json','/reported_ppl/32'),'raw',"authors' code-release reference value",'100%','reference value, not our measurement')
row('5.6888',(NM,'/20/paper_denominator_ppl'),'raw','same, 20 dense layers',(NM,'/20/suffix_kept'),'reproduction of saved reference 5.6903')
row('5.6903',(MC+'wikitext-reference.json','/reported_ppl/20'),'raw',"authors' code-release reference value",'49.63%','reference value, not our measurement')
row('1.30',(RQ,'/planner/overall/relative_ppl_increase'),'pct','Qwen whole-document, relaxed config (dense down proj, larger dense prefix)',(RQ,'/planner/overall/routed_kept'))
row('50.76',(RQ,'/planner/overall/routed_kept'),'pct','same','(retention)')
row('181',(RQ,'/neuro/overall/relative_ppl_increase'),'pct','same, our NeuroPrefetcher reimplementation',(RQ,'/neuro/overall/routed_kept'))
row('50.00',(RQ,'/neuro/overall/routed_kept'),'pct','same','(retention)')
# ---- Table 2 / 5.2
row('81.18',(HS%'dense','/accuracy_normalized'),'pct',HSP,'100%','single seed')
for f,v,rv,dl in ((HSM%'50','79.49','47.65','1.69'),(HSM%'40','79.11','39.66','2.07'),(HS%'griffin-50','74.10','48.05','7.08'),(HS%'griffin-40','70.93','40.04','10.25'),(HS%'neuro-50','60.92','47.97','20.25'),(HS%'neuro-40','52.55','40.04','28.63')):
    row(v,(f,'/accuracy_normalized'),'pct',HSP,(f,'/suffix_kept'),'single seed')
    row(rv,(f,'/suffix_kept'),'pct',HSP,'(retention)','single seed')
    row(dl,d('100*(v[1]-v[0])',(f,'/accuracy_normalized'),(HS%'dense','/accuracy_normalized')),'raw',HSP+'; drop from dense',(f,'/suffix_kept'),'derived difference, single seed')
for i,v,rv,dl in ((0,'77.09','29.70','4.09'),(1,'73.99','19.74','7.19')):
    row(v,(HSL,f'/results/{i}/accuracy_normalized'),'pct',HSP,(HSL,f'/results/{i}/suffix_kept'),'single seed')
    row(rv,(HSL,f'/results/{i}/suffix_kept'),'pct',HSP,'(retention)','single seed')
    row(dl,d('100*(v[1]-v[0])',(HSL,f'/results/{i}/accuracy_normalized'),(HS%'dense','/accuracy_normalized')),'raw',HSP+'; drop from dense',(HSL,f'/results/{i}/suffix_kept'),'derived difference, single seed')
for v,ptr in (('0.11','/paired/planner-20-vs-griffin-50/gap_percentage_points'),('0.77','/paired/planner-20-vs-griffin-50/ci95_percentage_points/0'),('0.53','/paired/planner-20-vs-griffin-50/ci95_percentage_points/1')):
    row(v,(HSL,ptr),'raw',HSP+'; planner-20 minus GRIFFIN-48, percentage points',(HSL,'/results/1/suffix_kept'),'paired example bootstrap, single seed')
row('2.4',d('v[0]/v[1]',(HS%'griffin-50','/suffix_kept'),(HSL,'/results/1/suffix_kept')),'raw',HSP+'; retention ratio GRIFFIN-48 / planner-20','-','derived ratio')
# ---- budget curve
for b,v,rv,ag,lo,hi in (('50','4.93','51.36',None,None,None),('40','7.80','41.49',None,'7.25','8.32')):
    f=PH%(b,'evaluation'); row(v,(f,O+'relative_ppl_increase'),'pct',WD,(f,O+'routed_kept')); row(rv,(f,O+'routed_kept'),'pct',WD,'(retention)')
    if lo: row(lo,(f,O+'relative_ppl_increase_ci95/0'),'pct',WD,(f,O+'routed_kept')); row(hi,(f,O+'relative_ppl_increase_ci95/1'),'pct',WD,(f,O+'routed_kept'))
for b,v,rv,ag,lo,hi in (('budget-30','12.50','31.48','85.56',None,None),('budget-20','21.31','21.46','81.97','20.16','22.47'),('calibrated-25-from-30','16.56','26.61',None,None,None)):
    f=LO%(b,'evaluation'); row(v,(f,O+'relative_ppl_increase'),'pct',WD,(f,O+'routed_kept')); row(rv,(f,O+'routed_kept'),'pct',WD,'(retention)')
    if ag: row(ag,(f,O+'top1_agreement'),'pct',WD,(f,O+'routed_kept'))
    if lo: row(lo,(f,O+'relative_ppl_increase_ci95/0'),'pct',WD,(f,O+'routed_kept')); row(hi,(f,O+'relative_ppl_increase_ci95/1'),'pct',WD,(f,O+'routed_kept'))
VAL='Mistral validation curve, 24 validation docs'
for path,v in ((LO%('budget-30/check-05000000','validation'),'12.8'),(LO%('budget-30/check-10000000','validation'),'12.1'),(LO%('budget-20/check-05000000','validation'),'23.2'),(LO%('budget-20/check-10000000','validation'),'21.5'),
               (PH%('40','check-10000000/validation'),'7.48'),(PH%('40','check-15000000/validation'),'7.28'),(PH%('50','check-20000000/validation'),'8.04'),(PH%('50','check-25000000/validation'),'4.85')):
    row(v,(path,O+'relative_ppl_increase'),'pct',VAL,(path,O+'routed_kept'),'validation, single seed')
row('50.1',d('100*0.78*(30/32)*(1-v[0])',(LO%('budget-30','evaluation'),O+'routed_kept')),'raw','share of all Mistral weights excluded at 31.48% retention (printed "roughly half")','-','derived: MLP share 0.778 x 30/32 sparse layers x (1 - retention)')
# ---- Qwen 5.4 and Table 4
for b,v,rv in (('50','8.60','51.62'),('40','15.70','41.14')):
    f=QP%('autoregressive',b,'evaluation'); row(v,(f,O+'relative_ppl_increase'),'pct',QWD,(f,O+'routed_kept')); row(rv,(f,O+'routed_kept'),'pct',QWD,'(retention)')
for v,b in (('8.6','50'),('15.7','40')):
    f=QP%('autoregressive',b,'evaluation'); row(v,(f,O+'relative_ppl_increase'),'pct',QWD,(f,O+'routed_kept'))
for arm,b,v,rv in (('one_shot_hidden','50','90.47','49.9'),('one_shot_hidden','40','113.55','40.0'),('one_shot_history','50','12.37','51.1'),('one_shot_history','40','22.20','40.9')):
    f=QP%(arm,b,'evaluation'); row(v,(f,O+'relative_ppl_increase'),'pct',QWD,(f,O+'routed_kept')); row(rv,(f,O+'routed_kept'),'pct',QWD,'(retention)')
for b,v,rv in (('50','25.64','50.8'),('40','38.36','40.8')):
    f=QP%('autoregressive',b,'no-previous-matched'); row(v,(f,O+'relative_ppl_increase'),'pct',QWD+', previous path removed at evaluation',(f,O+'routed_kept')); row(rv,(f,O+'routed_kept'),'pct',QWD,'(retention)')
row('51.6',(QP%('autoregressive','50','evaluation'),O+'routed_kept'),'pct',QWD,'(retention)'); row('41.1',(QP%('autoregressive','40','evaluation'),O+'routed_kept'),'pct',QWD,'(retention)')
for arm,b,v,rv in (('one_shot_history','50','12.07','51.1'),('one_shot_history','40','21.43','41.4'),('autoregressive','50','8.67','51.4'),('autoregressive','40','15.40','41.1')):
    f=Q2%(arm,b); row(v,(f,O+'relative_ppl_increase'),'pct',QWD+', second training seed',(f,O+'routed_kept'),'document bootstrap, seed 2')
    row(rv,(f,O+'routed_kept'),'pct',QWD+', seed 2','(retention)','seed 2')
for v,expr,refs in (('3.8','100*(v[0]-v[1])',((QP%('one_shot_history','50','evaluation'),O+'relative_ppl_increase'),(QP%('autoregressive','50','evaluation'),O+'relative_ppl_increase'))),
                    ('6.5','100*(v[0]-v[1])',((QP%('one_shot_history','40','evaluation'),O+'relative_ppl_increase'),(QP%('autoregressive','40','evaluation'),O+'relative_ppl_increase'))),
                    ('3.4','100*(v[0]-v[1])',((Q2%('one_shot_history','50'),O+'relative_ppl_increase'),(Q2%('autoregressive','50'),O+'relative_ppl_increase'))),
                    ('6.0','100*(v[0]-v[1])',((Q2%('one_shot_history','40'),O+'relative_ppl_increase'),(Q2%('autoregressive','40'),O+'relative_ppl_increase')))):
    row(v,d(expr,*refs),'raw',QWD+'; one-shot minus planner, percentage points','see rows above','derived difference')
row('62.16',(NQ,'/results/0.5162/overall/student_perplexity'),'raw',QWD+', our NeuroPrefetcher reimplementation',(NQ,'/results/0.5162/overall/routed_kept'))
row('368.88',(NQ,'/results/0.4000/overall/student_perplexity'),'raw',QWD+', our NeuroPrefetcher reimplementation',(NQ,'/results/0.4000/overall/routed_kept'))
row('5.89',(NQ,'/results/0.5000/overall/dense_perplexity'),'raw',QWD+', dense','100%','-')
# ---- Table 5
for b in ('50','40'):
    for arm,v in {'50':(('baseline','7.71'),('control','6.48'),('warm_prior','6.02'),('prior_first','44.23')),'40':(('baseline','13.77'),('control','12.33'),('warm_prior','11.12'),('prior_first','70.09'))}[b]:
        row(v,(SP,f'/results/{b}/{arm}/overall/relative_ppl_increase'),'pct','Qwen continuation, 64-token dense prompt, 63 test docs',(SP,f'/results/{b}/{arm}/overall/routed_kept'))
row('51.24',(SP,'/results/50/warm_prior/overall/routed_kept'),'pct','same','(retention)'); row('51.53',(SP,'/results/50/control/overall/routed_kept'),'pct','same','(retention)'); row('40.95',(SP,'/results/40/control/overall/routed_kept'),'pct','same','(retention)')
row('0.46',d('100*(v[0]-v[1])',(SP,'/results/50/control/overall/relative_ppl_increase'),(SP,'/results/50/warm_prior/overall/relative_ppl_increase')),'raw','same; control minus prior, pp','see rows','derived difference, single seed')
row('1.21',d('100*(v[0]-v[1])',(SP,'/results/40/control/overall/relative_ppl_increase'),(SP,'/results/40/warm_prior/overall/relative_ppl_increase')),'raw','same; control minus prior, pp','see rows','derived difference, single seed')
for v,ptr in (('7.71','/results/50/baseline/overall/relative_ppl_increase'),('6.48','/results/50/control/overall/relative_ppl_increase'),('6.02','/results/50/warm_prior/overall/relative_ppl_increase')): pass
# ---- Table 3
TQ='Qwen whole-document, 63 test docs'
row('2,259',(NG,'/results/fixed-50/overall/relative_ppl_increase'),'pct',TQ,(NG,'/results/fixed-50/overall/routed_mlp_retained')); row('3,653',(NG,'/results/fixed-40/overall/relative_ppl_increase'),'pct',TQ,(NG,'/results/fixed-40/overall/routed_mlp_retained'))
for v,f in (('23.16',(EM+'e2e-pilot-v1/evaluation-0000000.json','/conditions/topk-50/overall/')),('38.32',(EM+'e2e-pilot-v1/evaluation-0000000.json','/conditions/topk-40/overall/')),
            ('18.85',(EM+'e2e-hidden-pilot-v1/evaluation-0000000.json','/conditions/topk-50/overall/')),('30.48',(EM+'e2e-hidden-pilot-v1/evaluation-0000000.json','/conditions/topk-40/overall/'))):
    row(v,(f[0],f[1]+'relative_ppl_increase'),'pct',TQ+', imitation only',(f[0],f[1]+'routed_kept'))
for v,p in (('18.9',(EM+'e2e-hidden-pilot-v1/evaluation-0000000.json','/conditions/topk-50/overall/relative_ppl_increase')),('23.2',(EM+'e2e-pilot-v1/evaluation-0000000.json','/conditions/topk-50/overall/relative_ppl_increase'))): row(v,p,'pct',TQ+', imitation only','50.00%')
for v,f in (('3.86',EM+'e2e-full-v1/budget-50/evaluation-10000000.json'),('6.49',EM+'e2e-full-v1/budget-40/evaluation-10000000.json'),('2.64',EM+'e2e-hidden-full-v1/budget-50/evaluation-10000000.json'),('4.75',EM+'e2e-hidden-full-v1/budget-40/evaluation-10000000.json')):
    row(v,(f,O+'relative_ppl_increase'),'pct',TQ+', end to end (3M pilot + 10M continuation)',(f,O+'routed_kept'))
row('2.6',(EM+'e2e-hidden-full-v1/budget-50/evaluation-10000000.json',O+'relative_ppl_increase'),'pct',TQ,(EM+'e2e-hidden-full-v1/budget-50/evaluation-10000000.json',O+'routed_kept'))
row('3.9',(EM+'e2e-full-v1/budget-50/evaluation-10000000.json',O+'relative_ppl_increase'),'pct',TQ,(EM+'e2e-full-v1/budget-50/evaluation-10000000.json',O+'routed_kept'))
row('6.0',d('v[0]/v[1]',(EM+'e2e-pilot-v1/evaluation-0000000.json','/conditions/topk-50/overall/relative_ppl_increase'),(EM+'e2e-full-v1/budget-50/evaluation-10000000.json',O+'relative_ppl_increase')),'raw',TQ+'; imitation / e2e ratio, path predictor (printed "6 to 7 times")','-','derived ratio')
row('7.1',d('v[0]/v[1]',(EM+'e2e-hidden-pilot-v1/evaluation-0000000.json','/conditions/topk-50/overall/relative_ppl_increase'),(EM+'e2e-hidden-full-v1/budget-50/evaluation-10000000.json',O+'relative_ppl_increase')),'raw',TQ+'; imitation / e2e ratio, hidden predictor (printed "6 to 7 times")','-','derived ratio')
for v,k in (('0.96','oracle-50'),('2.39','oracle-40'),('0.99','teal_mlp-50'),('2.05','teal_mlp-40'),('1.03','wina_mlp-50'),('2.64','wina_mlp-40')):
    row(v,(NG,f'/results/{k}/overall/relative_ppl_increase'),'pct',TQ+', reference selector with activations known',(NG,f'/results/{k}/overall/routed_mlp_retained'))
# ---- 6.3 granularity
row('1.12',(SC2%'calibration-results','/results/mlp-g1-magnitude-k50-none/overall/relative_ppl_increase'),'pct',TQ+', single neurons, magnitude, activations known','50% exact top-k per layer (no realized field recorded)')
row('0.0136',(SC2%'calibration-results','/results/mlp-g1-magnitude-k50-none/overall/teacher_kl_nats'),'raw','same','same row')
row('94.76',(SC2%'calibration-results','/results/mlp-g1-magnitude-k50-none/overall/top1_agreement'),'pct','same','same row')
for v,g in (('22.5','16'),('28.3','32'),('37.1','128')):
    row(v,(SC2%'grid-results',f'/results/mlp-g{g}-contribution-k50-none/overall/relative_ppl_increase'),'pct',TQ+f', {g}-neuron groups, exact-contribution selection','50% exact top-k groups per layer (no realized field recorded)')
row('1.02',(FG+'results.json','/paired_comparisons/g32-k50/fresh_vs_old_relative_ppl_change'),'pct',TQ+' screening windows; fresh vs old 32-groups, relative PPL change','50% groups','paired, single seed')
row('22',(FG+'results.json','/results/g16-k50-fresh/overall/relative_ppl_increase'),'pct','same; fresh 16-groups','50% groups'); row('27',(FG+'results.json','/results/g32-k50-fresh/overall/relative_ppl_increase'),'pct','same; fresh 32-groups','50% groups')
import glob as _g
_fs=sorted(_g.glob(os.path.join(ROOT,FG+'groups-16-layer-*.json')))
row('98',d('100*sum(v)/len(v)',*[(os.path.relpath(f,ROOT),'/fraction_new_groups_crossing_old_128_boundaries') for f in _fs]),'raw','fresh 16-groups crossing old 128 boundaries, mean over 28 layers','-','derived mean (97.79%; 32-groups 97.94%)')
row('78.03',(NQ,'/results/0.5000/overall/student_perplexity'),'raw',TQ+', our NeuroPrefetcher reimplementation, predictor',(NQ,'/results/0.5000/overall/routed_kept'))
row('76.57',(NO,O+'student_perplexity'),'raw',TQ+', true nearest pattern, no predictor',(NO,O+'routed_kept'))
# ---- 6.4
for v,k in (('0.514','binary'),('0.641','hidden'),('0.517','magnitudes')):
    row(v,(PZ,f'/0.25/normalizations/{k}/normalized_skill/value'),'raw','Qwen phase-0 imitation probe, 128-neuron groups, 25% gated budget','25% groups','mean of 3 seeds')
CM='docs/e2e-matched-comparisons.json'
for v,p in (('1.22','/50/ppl_increase_difference_pp'),('0.82','/50/ppl_difference_ci95_pp/0'),('1.58','/50/ppl_difference_ci95_pp/1'),('1.74','/40/ppl_increase_difference_pp'),('1.22','/40/ppl_difference_ci95_pp/0'),('2.27','/40/ppl_difference_ci95_pp/1')):
    row(v,(CM,p),'raw',TQ+'; path minus hidden e2e predictor, pp','~50.6% / ~40.7%','paired document bootstrap, first seed')
# ---- 6.6
for arm,v in (('A','3.02'),('B','6.48'),('C','9.10'),('D','1,145.15')):
    f=HI%(arm,'evaluation-02000000'); row(v,(f,O+'relative_ppl_increase'),'pct',TQ+f', gate hierarchy arm {arm} at 2M tokens',(f,O+'routed_kept'))
f=HI%('D','native-final'); row('10.49',(f,O+'relative_ppl_increase'),'pct',TQ+', arm D at training-time threshold',(f,O+'routed_kept'))
for k,v in (('k75-none','14.1'),('k50-none','344'),('k75-edges','9.7'),('k50-edges','79')):
    row(v,(SC2%'grid-results',f'/results/attention-g128-contribution-{k}/overall/relative_ppl_increase'),'pct',TQ+f', attention heads {k}, activations known',('57.14% of heads (dense first/last two layers)' if 'edges' in k else k[1:3]+'% of heads, exact top-k'))
# ---- 3.3 hard gates
row('8.19',(R+'hard-gates--hard-gate-comparison-v1/raw/autoregressive/budget-50/evaluation.json',O+'relative_ppl_increase'),'pct',QWD+', deterministic gates from step 1',(R+'hard-gates--hard-gate-comparison-v1/raw/autoregressive/budget-50/evaluation.json',O+'routed_kept'))
row('1.06',(HG,'/hard_minus_control/ppl_difference_ci95_pp/0'),'raw',QWD+'; hard minus control, pp','see row','paired document bootstrap'); row('0.23',(HG,'/hard_minus_control/ppl_difference_ci95_pp/1'),'raw','same','see row','paired document bootstrap')
# ---- limitations
row('61.42',(NPC,'/neuro/50-training/accuracy/gate_top1'),'pct','Qwen NeuroPrefetcher reimplementation, pattern top-1 accuracy','-','single run'); row('48.81',(NPC,'/neuro/50-training/accuracy/up_top1'),'pct','same','-','single run')
# ---- Section 7
for k,v in (('gsm8k-dense','40.49'),('gsm8k-planner-30','15.16'),('gsm8k-planner-20','8.87'),('mmlu-dense','62.42'),('mmlu-planner-30','55.99'),('mmlu-planner-20','53.13')):
    row(v,(RS,f'/conditions/{k}/accuracy'),'pct','Mistral GSM8K 1,319 test 8-shot greedy / MMLU 14,042 test 5-shot',(RS,f'/conditions/{k}/routed_kept'),'single seed')
for k,v in (('gsm8k-planner-30','15.2'),('gsm8k-planner-20','10.2'),('mmlu-planner-30','13.2'),('mmlu-planner-20','8.4')):
    row(v,(RS,f'/conditions/{k}/routed_kept'),'pct','same','(retention)','single seed')
for k,trip in (('gsm8k-planner-30',('25.3','28.1','22.4')),('gsm8k-planner-20',('31.6','34.4','29.0')),('mmlu-planner-30',('6.4','7.2','5.7')),('mmlu-planner-20',('9.3','10.1','8.5'))):
    for v,p in zip(trip,('/paired_accuracy_gap_vs_dense','/paired_accuracy_gap_ci95/0','/paired_accuracy_gap_ci95/1')):
        row(v,(RS,f'/conditions/{k}{p}'),'pct','same; gap to dense, points',(RS,f'/conditions/{k}/routed_kept'),'paired bootstrap, single seed')
row('11.2',(RS,'/conditions/gsm8k-planner-30/quality/relative_ppl_increase'),'pct','GSM8K reference solutions, teacher forced, "30%" planner',(RS,'/conditions/gsm8k-planner-30/quality/routed_kept'),'single seed')
row('20.8',(RS,'/conditions/gsm8k-planner-30/quality/routed_kept'),'pct','same','(retention)','single seed')
LTP='Mistral decode timing, 8 WikiText-2 blocks, 64 tokens, batch 1, A100 80GB'
row('27.2',(LT,'/gpu-100-dense/inference_tokens_per_second'),'raw',LTP+', GPU dense','100%','tokens/s'); row('1.03',(LT,'/cpu-100-dense/inference_tokens_per_second'),'raw',LTP+', CPU-resident MLP dense','100%','tokens/s')
for k,v,rv in (('gpu-30-masked','0.47','44.2'),('gpu-30-gather','0.44','44.1'),('gpu-20-gather','0.44','30.8'),('cpu-30-gather','0.40','44.1'),('cpu-20-gather','0.61','30.7')):
    row(v,(LT,f'/{k}/speedup_vs_dense'),'raw',LTP+f', {k}',(LT,f'/{k}/routed_kept'),'paired block bootstrap'); row(rv,(LT,f'/{k}/routed_kept'),'pct','same','(retention)')
for k,a,b in (('cpu-30-gather','0.36','0.44'),('cpu-20-gather','0.52','0.73')):
    row(a,(LT,f'/{k}/speedup_ci95/0'),'raw',LTP,(LT,f'/{k}/routed_kept'),'paired block bootstrap'); row(b,(LT,f'/{k}/speedup_ci95/1'),'raw',LTP,(LT,f'/{k}/routed_kept'),'paired block bootstrap')
for k,v in (('cpu-100-dense','11.3'),('cpu-30-gather','5.4'),('cpu-20-gather','4.0')):
    row(v,(LT,f'/{k}/cpu_to_device_weight_payload_bytes_per_token'),'raw',LTP+' (GB/token = bytes/1e9)',(LT,f'/{k}/routed_kept'),'measured payload') if False else row(v,d('v[0]/1e9',(LT,f'/{k}/cpu_to_device_weight_payload_bytes_per_token')),'raw',LTP+', CPU-to-GPU payload GB/token',(LT,f'/{k}/routed_kept'),'measured')
row('52',d('100*(1-v[0]/v[1])',(LT,'/cpu-30-gather/cpu_to_device_weight_payload_bytes_per_token'),(LT,'/cpu-100-dense/cpu_to_device_weight_payload_bytes_per_token')),'raw','payload reduction','44.1%','derived')
row('65',d('100*(1-v[0]/v[1])',(LT,'/cpu-20-gather/cpu_to_device_weight_payload_bytes_per_token'),(LT,'/cpu-100-dense/cpu_to_device_weight_payload_bytes_per_token')),'raw','payload reduction','30.7%','derived')
row('4.9',d("max(max(r['max_absolute_logit_difference'] for r in x) for x in v)",*[(LTD%k,'') for k in ('gpu-30-gather','gpu-20-gather','cpu-30-gather','cpu-20-gather')]),'raw','gathered vs masked BF16 parity, max over all blocks and arms','-','derived max')
row('93',d("100*min(sum(r['top1_agreement_vs_same_policy_masked'] for r in x)/len(x) for x in v)",*[(LTD%k,'') for k in ('gpu-30-gather','gpu-20-gather','cpu-30-gather','cpu-20-gather')]),'raw','gathered vs masked top-1 agreement, lowest arm mean (printed 93–96%)','-','derived')
row('96',d("100*max(sum(r['top1_agreement_vs_same_policy_masked'] for r in x)/len(x) for x in v)",*[(LTD%k,'') for k in ('gpu-30-gather','gpu-20-gather','cpu-30-gather','cpu-20-gather')]),'raw','gathered vs masked top-1 agreement, highest arm mean','-','derived')
SSP='Mistral self-speculative greedy, 16 WikiText-2 blocks, 64 generated tokens'
for k,v,m in (('planner-20-k4','2.6','accepted_draft_tokens_per_verification'),('planner-30-k8','4.4','accepted_draft_tokens_per_verification'),('planner-20-k8','51','proposed_acceptance_fraction'),('planner-30-k4','71','proposed_acceptance_fraction'),('planner-20-k8','0.25','decode_speedup'),('planner-20-k4','0.32','decode_speedup')):
    row(v,(SS,f'/conditions/{k}/{m}'),'pct' if 'fraction' in m else 'raw',SSP+f', {k} {m} (range endpoint)',(SS,f'/conditions/{k}/draft_routed_kept'),'single seed')
# ---- costs
row('49.09',c('`results/mistral--mistral-phased-v1/README.md` posted charge $49.0877 (GPU $49.0117 + disk $0.0759)'),'raw','cost','-','posted charge')
row('7.1',d('(v[0]+v[1]+v[2])/3600',(R+'mistral--mistral-full-v1/raw/copy/completion.json','/seconds'),(R+'mistral--mistral-full-v1/raw/scheduled/completion.json','/seconds'),(R+'mistral--mistral-full-v1/raw/budget-50/progress.json','/seconds')),'raw','training GPU-hours, first allocation ($1.79/h from raw/cost-plan.json)','-','derived; no posted charge recorded')
row('24',d('(v[0]+v[1])/3600',(LO%('budget-30','completion'),'/seconds'),(LO%('budget-20','completion'),'/seconds')),'raw','training GPU-hours, 30% and 20% phases','-','derived')
row('50',d('2.09*(v[0]+v[1])/3600',(LO%('budget-30','completion'),'/seconds'),(LO%('budget-20','completion'),'/seconds')),'raw','estimated cost at $2.09/h (details/configs/mistral-budget-policy.json)','-','derived estimate, not posted')
row('0.13',(R+'qwen--followup-asset-audit-v1/cost.json','/estimated_cost'),'raw','lowest recorded Qwen run estimate','-','estimate')
row('8',(R+'qwen--planner-seed2-v2/cost.json','/estimated_cost'),'raw','highest recorded Qwen run estimate','-','estimate')
# ---- setup constants and arithmetic
row('34,000',(f'{MC}prefix-2-prompt-0/budget-50/documents/planner.json',O+'tokens'),'raw','Mistral scored positions, 63 test docs','-','count')
row('28,857',(QP%('autoregressive','50','evaluation'),O+'tokens'),'raw','Qwen scored positions, 63 test docs','-','count')
row('63',(QP%('autoregressive','50','evaluation'),O+'windows'),'raw','test documents','-','count')
row('1,319',(RS,'/conditions/gsm8k-dense/examples'),'raw','GSM8K test problems','-','count')
row('337,038,080',c('planner parameter count, Mistral; recomputed: (32+29+30)·14336·256 tables + 30·14336 head biases + 4096·256 W_h + 1,579,520 shared transformer + position/prev vectors'),'raw','architecture','-','-')
row('126,527,232',c('planner parameter count, Qwen; same breakdown with L=28, n=6144, hidden 2048'),'raw','architecture','-','-')
row('1.58M',c('shared 2-block transformer, width 256, FF 1024: 1,579,520 parameters'),'raw','architecture','-','-')
for v,desc in (('14.5','Mistral-7B parameters 7.24B x 2 bytes (BF16)'),('140','70B x 2 bytes'),('32','PCIe 4.0 x16 peak, external spec'),('7','fast PCIe 4.0 NVMe sequential read, external spec'),('2–3 TB/s','HBM bandwidth (A100 2.0 TB/s, H100 3.35 TB/s), external spec'),
               ('78','Mistral MLP share: 32·3·4096·14336 / 7.24B = 77.8%'),('half a second','14.48 GB / 32 GB/s = 0.45 s'),('two seconds','14.48 GB / 7 GB/s = 2.07 s'),
               ('5','planner multiply-adds per token / Mistral per-token compute = 4.7–5.3% depending on whether the 30 shared-transformer steps are counted'),('10','planner / Qwen per-token compute = 9.6% with shared-transformer steps counted')):
    row(v,c(desc),'raw','external fact or arithmetic','-','-')
# ---- rounded restatements
row('5.2',(P2_64%('50','planner'),O+'relative_ppl_increase'),'pct',CONT+' (abstract/intro/conclusion restatement of 5.21)',(P2_64%('50','planner'),O+'routed_kept'))
row('2.1',d('100*(v[1]-v[0])',(HSM%'40','/accuracy_normalized'),(HS%'dense','/accuracy_normalized')),'raw',HSP+'; drop at ~40% (restatement of 2.07)',(HSM%'40','/suffix_kept'),'derived, single seed')
row('1.7',d('100*(v[1]-v[0])',(HSM%'50','/accuracy_normalized'),(HS%'dense','/accuracy_normalized')),'raw',HSP+'; drop at ~48%',(HSM%'50','/suffix_kept'),'derived, single seed')
row('1.2',d('100*(v[0]-v[1])',(P2_64%('40','planner'),O+'routed_kept'),(P2_64%('40','griffin'),O+'routed_kept')),'raw',CONT+'; planner minus baseline retention at 40% (printed 1.1–1.2 pp)','-','derived')
row('1.1',d('100*(v[0]-v[1])',(P2_64%('50','planner'),O+'routed_kept'),(P2_64%('50','neuro'),O+'routed_kept')),'raw',CONT+'; planner minus NeuroPrefetcher retention at 50%','-','derived')
row('126.5',c('planner parameters, Qwen (126,527,232; see architecture row)'),'raw','architecture','-','-')
row('337.0',c('planner parameters, Mistral (337,038,080; see architecture row)'),'raw','architecture','-','-')
# ---- configuration constants (pointer-checked where a config exists)
CF='configs/mistral-planner-phased.json'; CQ='configs/planner-pilot.json'; CL='configs/mistral-lower-budgets.json'
for v,p in (('256','/width'),('2','/depth'),('4','/heads'),('0.01','/epsilon'),('1','/beta'),('2','/positive_weight'),('0.05','/dual_learning_rate'),('100','/maximum_alpha'),('1.0','/temperatures/0'),('0.1','/temperatures/1'),('512','/sequence_length'),('0.001','/plateau_minimum_kl_improvement')):
    row(v,(CF,p),'raw','Mistral planner config','-','configuration')
for v,p in (('25','/phase_tokens/50'),('15','/phase_tokens/40'),('10','/copy_tokens'),('2','/scheduled_tokens'),('8','/phase40_hard_start_tokens')):
    row(v,d('v[0]/1e6',(CF,p)),'raw','Mistral planner config (millions of tokens)','-','configuration')
row('1',d('v[0]/1e6',(CQ,'/scheduled_tokens')),'raw','Qwen planner config, scheduled sampling (millions)','-','configuration')
row('20',d('100*v[0]',(CQ,'/hard_fraction')),'raw','Qwen planner config, deterministic final fraction (%)','-','configuration')
row('10',d('v[0]/1e6',(CL,'/quality_checkpoint_tokens/2')),'raw','Mistral 30%/20% phases final tokens (millions)','-','configuration')
row('10,042',(HS%'dense','/examples'),'raw','HellaSwag validation examples','-','count')
row('14,042',(RS,'/conditions/mmlu-dense/examples'),'raw','MMLU test questions','-','count')
for v,desc in (('30 of 32','sparse layers: layers 3–32 of Mistral (first two dense)'),('14,336 / 4,096 / 32','Mistral-7B-v0.1 config (intermediate_size, hidden_size, num_hidden_layers), revision 27d67f1b'),
               ('6,144 / 2,048 / 28','Qwen3-1.7B-Base config (intermediate_size, hidden_size, num_hidden_layers)'),('75th / 16','strong-bit threshold: pathpred/path_signals.py quantile .75; scripts/mistral_planner.py 16 reserved windows'),
               ('1,024 (FF width)','planner feed-forward width 4×256, pathpred/path_model.py'),('512 patterns','NeuroPrefetcher K=512 centroids (Dhar et al., §4)'),
               ('24 / 15 / 24; 24 validation (8 per domain)','test and validation document manifests, docs/setup'),('128–512','test window length in Qwen tokens, test manifest'),
               ('64 / 512 / 1,024 / 1,536','prompt lengths and WikiText-2 window length, configs/mistral-long-prompt.json'),('1,024 (HellaSwag calibration)','unlabeled HellaSwag training examples for calibration, hellaswag-matched-v1 protocol'),
               ('0.59','A40 hourly rate in Qwen run records'),('95','confidence level of all intervals')):
    row(v,c(desc),'raw','configuration or setup constant','-','-')
# ---- abstract (rounded restatements)
row('49',(P2_64%('50','planner'),O+'routed_kept'),'pct',CONT+' (abstract: retention for the 5.2% result)','(retention)')
row('72',(P2_64%('50','griffin'),O+'relative_ppl_increase'),'pct',CONT+' (abstract)',(P2_64%('50','griffin'),O+'routed_kept'))
row('228',(P2_64%('50','neuro'),O+'relative_ppl_increase'),'pct',CONT+' (abstract)',(P2_64%('50','neuro'),O+'routed_kept'))
row('48',(HS%'griffin-50','/suffix_kept'),'pct',HSP+' (abstract: GRIFFIN retention)','(retention)','single seed')
row('20',(HSL,'/results/1/suffix_kept'),'pct',HSP+' (abstract: NT-20 retention 19.74%)','(retention)','single seed')
