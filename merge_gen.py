import numpy as np
from typing import Optional, List
import torch.nn as nn
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from tqdm import tqdm
from datasets import load_dataset
import torch
import json
from typing import Optional
from dataclasses import dataclass, field

from transformers import AutoTokenizer, HfArgumentParser, pipeline, DataCollatorForSeq2Seq



#####
# This script merges the output files into one file.
#####

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    dataset_path: Optional[str] = field(
        default="/home/xx/xw/rsf/rsf_hf_mistral_rsf_baseline/model0/data/gen_data",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
from collections import defaultdict


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

data_ret = defaultdict(list)

repeat_num_0 = 0

for j in range(1, 9):
    ds = load_dataset("json", data_files=script_args.dataset_path + str(j) + ".json", split="train",
                      field="instances")  # .select(range(500))
    for sample in ds:
        data_ret[sample['prompt']].append(sample['responses'][0])

gathered_data = []

ccnt = 0
for key in data_ret:
    assert len(data_ret[key]) == 8
    if len(list(set(data_ret[key]))) < 8:
        print(len(list(set(data_ret[key]))))
        ccnt += 1
    if len(list(set(data_ret[key]))) < 3:
        continue
    gathered_data.append({"prompt": key, "responses": data_ret[key]})

    

print(ccnt)


#print(len(gathered_data))


output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
with open(script_args.dataset_path + ".json", 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

