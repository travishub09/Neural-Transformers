"""Pin benchmark data and tokenization on CPU before evaluation preregistration."""
import hashlib,json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
PINS={'openai/gsm8k':'740312add88f781978c0658806c59bc2815b9866','cais/mmlu':'c30699e8356da336a370243923dbaf21066bb9fe'}
REV='27d67f1b5f57dc0953326b2601d68371d40ea8da'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def main():
 root=Path('data/followup-reasoning-v1');root.mkdir(parents=True,exist_ok=True)
 tok=AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1',revision=REV)
 tok.save_pretrained(root/'tokenizer');manifest={'sources':{},'tokenizer_id':'mistralai/Mistral-7B-v0.1','tokenizer_revision':REV,'selection_seed':20261122,'test_used_for_selection':False}
 for task,dataset,config,split,shots in [('gsm8k','openai/gsm8k','main','train',8),('mmlu','cais/mmlu','all','dev',5)]:
  train=load_dataset(dataset,config,revision=PINS[dataset],split=split);test=load_dataset(dataset,config,revision=PINS[dataset],split='test')
  if task=='gsm8k':examples=sorted(list(train),key=lambda x:digest([20261122,x['question']]))[:shots]
  else:examples=None
  backup=load_dataset(dataset,config,revision=PINS[dataset],split='validation') if task=='mmlu' else []
  def format_question(x,answer=False):
   if task=='gsm8k':return 'Question: '+x['question']+'\nAnswer:'+(' '+x['answer']+'\n\n' if answer else '')
   value=x['question']+'\n'+''.join(f'{letter}. {choice}\n' for letter,choice in zip('ABCD',x['choices']))+'Answer:'
   return value+(' '+ 'ABCD'[x['answer']]+'\n\n' if answer else '')
  by_subject={}
  for row in list(train)+list(backup):by_subject.setdefault(row.get('subject','gsm8k'),[]).append(row)
  rows=[];train_hashes={digest(x['question']) for x in train};test_hashes={digest(x['question']) for x in test}
  # Do not silently discard test questions. Record any overlap instead of
  # pretending named splits guarantee text-level disjointness.
  for i,x in enumerate(test):
   ex=examples if examples is not None else [z for z in by_subject[x['subject']] if digest(z['question']) not in test_hashes][:shots]
   assert len(ex)==shots
   assert x['question'] not in {z['question'] for z in ex},'Few-shot prompt leaks test question'
   prefix=''.join(format_question(z,True) for z in ex)
   if task=='mmlu':prefix='The following are multiple choice questions (with answers) about '+x['subject'].replace('_',' ')+'.\n\n'+prefix
   prompt=prefix+format_question(x);ids=tok.encode(prompt,add_special_tokens=False)
   row=dict(id=i,document_sha256=digest(x['question']),domain=x.get('subject',task),prompt=prompt,prompt_ids=ids,shots=shots,fewshot_sha256=[digest(z['question']) for z in ex],source=x)
   if task=='mmlu':
    full=[tok.encode(prompt+' '+letter,add_special_tokens=False) for letter in 'ABCD']
    assert all(z[:len(ids)]==ids and len(z)==len(ids)+1 for z in full),'Letter token boundary changed'
    row['choice_token_ids']=[z[-1] for z in full];row['gold']=int(x['answer'])
   else:row['gold']=x['answer'].split('####')[-1].strip()
   rows.append(row)
  path=root/(task+'.jsonl');path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
  manifest['sources'][task]=dict(dataset=dataset,config=config,revision=PINS[dataset],split='test',field=['question','answer'] if task=='gsm8k' else ['question','choices','answer','subject'],examples=len(rows),fewshot_split=split if task=='gsm8k' else 'dev preferred; replace global test-question overlaps from same-subject validation in source order',shots=shots,fewshot_ids=sorted({h for x in rows for h in x['fewshot_sha256']}),test_ids=[x['document_sha256'] for x in rows],fewshot_test_disjoint=not {h for x in rows for h in x['fewshot_sha256']}&test_hashes,all_source_split_question_hash_overlap=len(train_hashes&test_hashes),prompt_tokens=sum(len(x['prompt_ids']) for x in rows),max_prompt_tokens=max(len(x['prompt_ids']) for x in rows),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),truncation='none')
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({k:{a:v[a] for a in ['examples','prompt_tokens','max_prompt_tokens','fewshot_test_disjoint','all_source_split_question_hash_overlap']} for k,v in manifest['sources'].items()},indent=2))
if __name__=='__main__':main()
