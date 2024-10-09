from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from tqdm import tqdm


from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)
import torch
import os 
import time 

from trl.trainer.utils import generate

#import deepspeed
#from deepspeed.utils.timer import SynchronizedWallClockTimer

# Initialize the timer
#timers = SynchronizedWallClockTimer()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

@dataclass
class ScriptArguments:
    # model parameters
    model_name_or_path: Optional[str] = field(default="lomahony/pythia-160m-helpful-sft", metadata={"help": "the SFT model name"})
    # data parameters
    dataset_name: Optional[str] = field(default="trl-internal-testing/hh-rlhf-helpful-base-trl-style", metadata={"help": "the HF data path"})
    dataset_col: Optional[str] = field(default="prompt", metadata={"help": "the HF data path"})
    split: Optional[str] = field(default="train", metadata={"help": "the dataset split to use for generation"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the generation batch size"})
    max_prompt_length: Optional[int] = field(default=100, metadata={"help": "the maximum prompt length"})
    subset_size: Optional[int] = field(default=500, metadata={"help": "the number of samples to use from the dataset for debugging"})
    save_dataset_path: Optional[str] = field(default="sft_gen_dataset_v2_notVLLM", metadata={"help": "the path for saving the generated dataset"})
    # generation parameters
    max_new_tokens: Optional[int] = field(default=128, metadata={"help": "the maximum number of tokens generated per sample"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "the sampling temperature"})
    top_p: Optional[float] = field(default=0.98, metadata={"help": "top_p sampling argument"})
    top_k: Optional[int] = field(default=25, metadata={"help": "top_k sampling argument"})
    num_return_sequences: Optional[int] = field(default=64, metadata={"help": "the number of return sequences"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=False)
    



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator()
    
    # load sft policy
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path,torch_dtype=torch.bfloat16,trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # for generation
    tokenizer.padding_side = "left"

    # define gen_kwargs
    generation_kwargs = {
        "top_k": script_args.top_k,
        "top_p": script_args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": script_args.temperature,
        "max_new_tokens": script_args.max_new_tokens,
        "num_return_sequences": script_args.num_return_sequences,
    }

    #start_time=time.time()
    # load and preprocess the dataset
    dataset = load_dataset(script_args.dataset_name)[script_args.split]
    dataset = dataset.select(range(min(len(dataset), script_args.subset_size)))


    if script_args.sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    def tokenize_fn(samples):
        model_inputs = tokenizer(samples[script_args.dataset_col])

        return {
            **model_inputs,
        }

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=list(dataset.features))
    dataset = dataset.filter(lambda x: len(x["input_ids"])<script_args.max_prompt_length)
    #load_time = time.time() - start_time

    data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=script_args.max_prompt_length, pad_to_multiple_of=8)

    dataloader = DataLoader(dataset, batch_size=script_args.batch_size, shuffle=False, collate_fn=data_collator)

    model, dataloader = accelerator.prepare(model, dataloader)

    
    start_time=time.time()

    # generate responses from sft policy
    prompts, responses = generate(model, dataloader, tokenizer, accelerator, seed=0, **generation_kwargs)
    generate_time = time.time() - start_time

    generated_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})
    
    # save the generated dataset
    generated_dataset.save_to_disk(script_args.save_dataset_path)
    #print(f"Time taken to load and preprocess dataset: {load_time:.2f} seconds")
    print(f"Time taken to generate responses (GPU): {generate_time:.6f} seconds")
