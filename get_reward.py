import torch.distributed as dist
from transformers import pipeline, AutoTokenizer
import time
from torch.utils.data import DataLoader

import os

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

import torch.nn as nn
import torch
from typing import Optional, List
import numpy as np

tqdm.pandas()


#####
# This script takes a dataset as the input, where each sample is {"input": "the pormpt", "output": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"input": "the pormpt", "output": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
# Due to memory constraint, we will set the reward of the input+output that is longer than 800 tokens as -999999, which should be discarded in later processing. It should be at most ~2% samples that are discarded.
#####

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    dataset_name_or_path: Optional[str] = field(
        default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/iter1",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    max_length: Optional[int] = field(
        default=9999999999,
        metadata={"help": "the maximum length of the prompt"},
    )
    reward_name_or_path: Optional[str] = field(
        default="weqweasdas/RM-Gemma-7B",
        metadata={"help": "the name of the gold reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    train_micro_batch_size_per_gpu: Optional[int] = field(
        default=4,
        metadata={"help": "the batch size for inference"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
# AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = script_args.train_micro_batch_size_per_gpu

device = accelerator.device
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}
reward_model = script_args.reward_name_or_path
rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)
rm_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True
)


ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))
ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")

local_rank = Accelerator().local_process_index

data_size = len(ds['prompt'])

share = int(data_size / world_size)
ds = ds.select(np.arange(local_rank * share, (local_rank + 1)*share))


def get_reward(test_texts):
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards


data = []
top1_data = []
cnt = 0

def change_of_format(prom, resp):
    prom = prom.replace("<|user|>\n", "").replace(
        "</s>\n", "").replace("<|assistant|>\n", "")
    resp = resp.replace("<|assistant|>\n", "")
    final_resp = resp.split("<|user|>")[0]

    message = [
        {"role": "user", "content": prom},
        {"role": "assistant", "content": final_resp},
    ]
    return rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, "")



    #prom = prom.replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model", "")
    #message = [
    ##    {"role": "user", "content": prom},
     #   {"role": "assistant", "content": resp},
    #]
    #return rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, "")


zzz = 0
# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        #test_texts = [sample['prompt'].replace(
        #    "<s>", "") + tmp_output + '</s>' for tmp_output in sample['responses']]
        if zzz == 0:
            print(change_of_format(sample['prompt'], sample['responses'][0]))
            zzz += 1
        test_texts = [change_of_format(sample['prompt'], tmp_output)
                      for tmp_output in sample['responses']]
        rewards = get_reward(test_texts)
        data.append(
            {"prompt": sample['prompt'], "responses": sample['responses'], "rewards": rewards})

        idx = np.argmax(rewards)
        top1_data.append({"prompt": sample['prompt'], 'response': sample['responses'][idx]})
        cnt += 1
        if (cnt + 1) % 1000 == 0:
            print(cnt)


# Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list = [{}] * world_size

data_to_send = {
    'data': [[data[i]] for i in range(len(data))],
    'top1_data': top1_data,
}


dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []
gathered_top1_data = []


for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
    gathered_data.extend(tmp_data)
    gathered_top1_data.extend(all_process_list[i]['top1_data'])

all_rewards = [sample['rewards'][:8] for sample in gathered_data]
top1_scores = np.mean(np.max(all_rewards, axis=1))
mean_scores = np.mean(all_rewards)


if local_rank == 0:
    print("Collect {} data. mean score {} top1 score: {}".format(
        len(gathered_top1_data), mean_scores, top1_scores))
    output_eval_dataset = {}
    output_eval_dataset['type'] = 'text_only'
    output_eval_dataset['instances'] = gathered_data
    with open(script_args.output_dir + "/data_with_rewards.json", 'w', encoding='utf8') as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)

    output_eval_dataset = {}
    output_eval_dataset['type'] = 'text_only'
    output_eval_dataset['instances'] = gathered_top1_data
    with open(script_args.output_dir + "/top1_data.json", 'w', encoding='utf8') as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)

    with open(script_args.record_dir, 'a') as f:
        f.write(str(mean_scores) + "\t" + str(top1_scores) + "\n")
