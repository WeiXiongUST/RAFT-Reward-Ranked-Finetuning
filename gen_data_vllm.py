from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np
from typing import List
from transformers import HfArgumentParser, AutoTokenizer, Trainer
from tqdm import tqdm
from datasets import load_dataset, Dataset
import torch
import json
from typing import Optional
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    url: Optional[str] = field(
        default="http://localhost",
        metadata={"help": "url of the model response"},
    )
    tokenizer: Optional[str] = field(
        default="gpt2",
        metadata={"help": "the tokenizer to use"},
    )
    ports: List[str] = field(default_factory=lambda: ["8000"], metadata={
                             "help": "ports of the model response"})
    dataset_name_or_path: Optional[str] = field(
        default=".json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default=".json",
        metadata={"help": "the location of the output file"},
    )
    bos_format: Optional[str] = field(
        # default="<start_of_turn>model\n",
        default="",
        metadata={"help": "the format of the beginning of the sentence"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=10000,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=1500,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="prompt",
        metadata={"help": "the key of the dataset"},
    )
    max_workers: Optional[int] = field(
        default=1024,
        metadata={"help": "the number of workers"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ds_dir = script_args.dataset_name_or_path
output_dir = script_args.output_dir
K = script_args.K
ports = script_args.ports

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer)
###
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def query_model(prompt, args, port):
    json = {
        **args, "prompt": prompt,
    }
    response = requests.post(
        url=script_args.url+":"+str(port)+"/generate",
        json=json)
    response_json = response.json()
    return [response_json['text'][i][len(prompt):] for i in range(len(response_json['text']))]


default_args = {
    "use_beam_search": script_args.use_beam_search,
    "n": script_args.K,
    "temperature": script_args.temperature,
    "max_tokens": script_args.max_new_tokens,
    "seed": script_args.seed,
    "top_p": 1.0,
    "top_k": -1,
    "stop_token_ids": [tokenizer.eos_token_id],
    #"stop": ["<|user|>", "</s>"]
}


ds = load_dataset("json", data_files=ds_dir, split="train",
                  field="instances")#.select(range(1000))

print(ds)

# use tokenizer.apply_template to apply the template to the prompt
ds = ds.map(lambda x: {"prompt": tokenizer.apply_chat_template(
    x[script_args.dataset_key], tokenize=False, add_generation_prompt=True)})


with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
    result = [executor.submit(query_model, ds[i]["prompt"],
                              default_args, ports[i % len(ports)]) for i in range(len(ds))]
    # use tqdm to show progress
    for _ in tqdm(as_completed(result), total=len(result)):
        pass

    responses = [r.result() for r in result]


gathered_data = []
all_prompts_ = ds[script_args.dataset_key]
for i in range(len(ds)):
    tmp_data = {"prompt": all_prompts_[i], "responses": responses[i]}
    gathered_data.append(tmp_data)

output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
print("I collect ", len(gathered_data), "samples")


with open(output_dir, 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

