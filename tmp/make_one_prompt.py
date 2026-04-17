from synthrlvl.task import TaskBuilder
from synthrlvl.types import TaskConfig, TemplateName, PrefillMode, StepRange
cfg = TaskConfig(template=TemplateName.LOGIC,prefill=PrefillMode.NONE,distractor_ratio=0.5,train_steps=StepRange(1,10),val_steps=StepRange(1,1),seed=3407)
s = TaskBuilder(cfg).build_samples(1, train=False)[0]
open('/home/hpc/c107fa/c107fa12/synthetic-RLVL/tmp/one_prompt_step1.txt','w').write(s.prompt)
print(s.prompt)
