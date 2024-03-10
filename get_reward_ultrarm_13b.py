import torch.distributed as dist
import numpy as np
from typing import Optional, List
import torch.nn as nn
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from tqdm import tqdm
from datasets import load_dataset
import torch
import json
from accelerate import Accelerator
from typing import Optional
from dataclasses import dataclass, field
import os
from torch.utils.data import DataLoader
import time
from transformers import pipeline, AutoTokenizer

from transformers import AutoTokenizer, HfArgumentParser, pipeline, DataCollatorForSeq2Seq
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer

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
        default="openbmb/UltraRM-13b",
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

device = accelerator.device
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}
reward_model = script_args.reward_name_or_path


class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(
            self.config.hidden_size, 1, bias=False)

    def forward(  # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        rewards = torch.gather(rewards, 1, ends)

        return rewards


device = accelerator.device
# device = 0
reward_model_name = "openbmb/UltraRM-13b"
tokenizer = LlamaTokenizer.from_pretrained(
    reward_model_name)
model = LlamaRewardModel.from_pretrained(
    reward_model_name).to(torch.bfloat16)
model = model.to(device)


ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))
ds = load_dataset("json", data_files=ds_dir, split="train",
                  field="instances")#.select(range(500))

local_rank = Accelerator().local_process_index

data_size = len(ds['prompt'])

share = int(data_size / world_size)
ds = ds.select(np.arange(local_rank * share, (local_rank + 1)*share))
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


model, opt = accelerator.prepare(model, optimizer)


def get_reward(texts):
    scores = []
    for txt in texts:
        inputs = tokenizer(txt, return_tensors="pt").to(device)
        if False:
            pass
        # if len(inputs['input_ids']) > 800:
        #    chosen_reward = -999999
        else:
            with torch.no_grad():
                chosen_reward = model(**inputs).item()
        scores.append(chosen_reward)
        # del inputs
    return scores


def change_of_format(prom, resp):
    prom = prom.replace("<|user|>\n", "").replace(
        "</s>\n", "").replace("<|assistant|>\n", "")
    resp = resp.replace("<|assistant|>\n", "")
    final_resp = resp.split("<|user|>")[0]

    return "Human: " + prom + "\nAssistant: " + final_resp


data = []
top1_data = []
cnt = 0

#print(ds[0])
# tqdm is used to show the progress bar
zzz = 0
with torch.no_grad():
    for sample in tqdm(ds):
        test_texts = [change_of_format(sample['prompt'], tmp_output)
                      for tmp_output in sample['responses']]
        rewards = get_reward(test_texts)
        data.append(
            {"prompt": sample['prompt'], "response": sample['responses'], "rewards": rewards})

        idx = np.argmax(rewards)
        if "<|user|>" in sample['responses'][idx] or "<|assistant|>" in sample['responses'][idx]:
            continue
        top1_data.append(
            {"prompt": sample['prompt'], 'response': sample['responses'][idx]})
        cnt += 1
        if (cnt + 1) % 500 == 0:
            print(cnt, local_rank)


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

all_rewards = [sample['rewards'] for sample in gathered_data]
top1_scores = np.mean(np.max(all_rewards, axis=1))
mean_scores = np.mean(all_rewards)


if local_rank == 0:
    print("Collect {} data. mean score {} top1 score: {}".format(
        len(gathered_top1_data), mean_scores, top1_scores))
    print(script_args.output_dir)
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
