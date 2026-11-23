import torch
from scripts.followup_reasoning import number,first_prediction,generate

class Fake:
 def __init__(self):self.model=torch.nn.Linear(1,1);self.reset()
 def reset(self):self.i=0
 def prefill(self,ids,valid):self.prefill_ids=ids;self.prefill_valid=valid
 def step(self,ids,valid):
  self.final_input=ids.clone();self.i+=1
  z=torch.zeros(2,5);z[0,2]=10;z[1,2 if self.i==3 else 3]=10
  return z
 def masks(self):return torch.tensor([[[True,False]],[[True,False]]])
class Tokenizer:
 eos_token_id=2
 def decode(self,x,**kwargs):return ' '.join(map(str,x))

def rows():return [dict(id=i,document_sha256=str(i),prompt_ids=p,gold='3') for i,p in enumerate([[1,4,1],[1,3,4,1]])]
def test_final_prompt_input_is_sparse_for_single_letter_prediction():
 p=Fake();z,valid,ret=first_prediction(p,rows())
 assert p.prefill_ids.tolist()==[[0,1,4],[1,3,4]]
 assert p.final_input.tolist()==[[1],[1]]
 assert p.i==1 and ret.tolist()==[.5,.5]
def test_finished_examples_do_not_accumulate_retention():
 result=generate(Fake(),rows(),Tokenizer(),8)
 assert [x['scored_executions'] for x in result]==[1,3]
 assert [x['kept_sum'] for x in result]==[.5,1.5]
def test_numeric_extraction_and_no_answer():
 assert number('work 42; #### 1,200.00')==number('1200',False)
 assert number('Only intermediate number 17') is None
 assert number('#### 2\n#### -3.5')=='-3.5'
