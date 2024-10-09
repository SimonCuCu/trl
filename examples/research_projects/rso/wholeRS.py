import os
import time
import random
import warnings
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from tqdm import tqdm
import pandas as pd
import sys

from accelerate import Accelerator, DistributedType
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
)

from trl.trainer.utils import generate, conduct_rejection_sampling, compute_reward_score

import os
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
    reward_model_name_or_path: Optional[str] = field(default="RLHFlow/ArmoRM-Llama3-8B-v0.1", metadata={"help": "the reward model name"})

    # data parameters
    dataset_name: Optional[str] = field(default="trl-internal-testing/hh-rlhf-helpful-base-trl-style", metadata={"help": "the HF data path"})
    dataset_col: Optional[str] = field(default="prompt", metadata={"help": "the HF data path"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the generation batch size"})
    batch_size_scoring: Optional[int] = field(default=32, metadata={"help": "the scoring batch size"})
    max_prompt_length: Optional[int] = field(default=100, metadata={"help": "the maximum prompt length"})
    subset_size: Optional[int] = field(default=1000, metadata={"help": "the number of samples to use from the dataset for debugging"})
    save_rewarded_dataset_path: Optional[str] = field(default="rewarded_dataset_tobeRS", metadata={"help": "the path for saving the dataset with rewards"})
    save_dataset_path_tobeDPO: Optional[str] = field(default="dataset_to_be_DPO", metadata={"help": "the path for saving the dataset to be finetuned by DPO"})

    # generation parameters
    max_new_tokens: Optional[int] = field(default=128, metadata={"help": "the maximum number of tokens generated per sample"})
    top_p: Optional[float] = field(default=0.98, metadata={"help": "top_p sampling argument"})
    top_k: Optional[int] = field(default=50, metadata={"help": "top_k sampling argument"})
    num_return_sequences: Optional[int] = field(default=32, metadata={"help": "the number of return sequences"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "the sampling temperature"})

    # rejection sampling 
    num_samples: Optional[int] = field(default=8, metadata={"help": "the number of samples to keep after rejection sampling"})
    beta: Optional[int] = field(default=.5, metadata={"help": "Beta parameter for rejection sampling"})
    ranking_method: Optional[str] = field(default="first_round", metadata={"help": "ranking method: 'first_round' or 'tournament'"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False)
    split: Optional[str] = field(default="train", metadata={"help": "the dataset split to use for generation"})


def first_round_ranking(responses: List[str], rewards: List[float]) -> Tuple[List[str], List[str]]:
    """Conducts first round ranking."""
    chosen = []
    rejected = []
    responses = [(response, reward) for response, reward in zip(responses, rewards)]
    
    def pick(responses):
        selected = random.randrange(len(responses))
        return responses.pop(selected)
    
    while responses:
        selected1 = pick(responses)
        selected2 = pick(responses)
        if selected1[1] > selected2[1]:
            chosen.append(selected1[0])
            rejected.append(selected2[0])
        else:
            chosen.append(selected2[0])
            rejected.append(selected1[0])
            
    return chosen, rejected

def tournament_ranking(responses: List[str], rewards: List[float]):
    """Conducts tournament ranking."""
    sorted_responses = [response for _, response in sorted(zip(rewards, responses), reverse=True)]
    chosen = [sorted_responses[i] for i in range(0, len(responses), 2)]
    rejected = [sorted_responses[i] for i in range(1, len(responses), 2)]
    return chosen, rejected

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator()

    # Load SFT model and tokenizer
    sft_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    sft_tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if sft_tokenizer.pad_token_id is None:
        sft_tokenizer.pad_token_id = sft_tokenizer.eos_token_id
    sft_tokenizer.padding_side = "left"

    # Load Reward model and tokenizer
    reward_model = AutoModelForSequenceClassification.from_pretrained(script_args.reward_model_name_or_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model_name_or_path)
    reward_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    reward_model.resize_token_embeddings(len(reward_tokenizer))
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    # Define generation kwargs
    generation_kwargs = {
        "top_k": script_args.top_k,
        "top_p": script_args.top_p,
        "do_sample": True,
        "pad_token_id": sft_tokenizer.pad_token_id,
        "max_new_tokens": script_args.max_new_tokens,
        "num_return_sequences": script_args.num_return_sequences,
        "temperature": script_args.temperature
    }

    # Load and preprocess the dataset
    dataset = load_dataset(script_args.dataset_name)[script_args.split]
    dataset = dataset.select(range(min(len(dataset), script_args.subset_size)))

    if script_args.sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    def tokenize_sft_fn(samples):
        model_inputs = sft_tokenizer(samples[script_args.dataset_col])

        return {
            **model_inputs,
        }

    dataset = dataset.map(tokenize_sft_fn, batched=True, remove_columns=list(dataset.features))
    dataset = dataset.filter(lambda x: len(x["input_ids"]) < script_args.max_prompt_length)

    data_collator = DataCollatorForSeq2Seq(sft_tokenizer, max_length=script_args.max_prompt_length, pad_to_multiple_of=8)
    dataloader = DataLoader(dataset, batch_size=script_args.batch_size, shuffle=False, collate_fn=data_collator)
    sft_model, dataloader = accelerator.prepare(sft_model, dataloader)

    # Generate responses from SFT policy
    start_time = time.time()
    prompts, responses = generate(sft_model, dataloader, sft_tokenizer, accelerator, seed=0, **generation_kwargs)
    generate_time = time.time() - start_time

    # Combine prompts and responses into a dataset
    generated_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})

    # Tokenize for reward model
    def tokenize_reward_fn(samples):
        text = [prompt + " " + response for prompt, response in zip(samples["prompt"], samples["response"])]
        return reward_tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    reward_dataset = generated_dataset.map(tokenize_reward_fn, batched=True, remove_columns=list(generated_dataset.features))
    data_collator = DataCollatorWithPadding(reward_tokenizer)
    dataloader = DataLoader(reward_dataset, batch_size=script_args.batch_size_scoring, shuffle=False, collate_fn=data_collator)
    reward_model, dataloader = accelerator.prepare(reward_model, dataloader)

    # Compute rewards
    start_time = time.time()
    rewards = compute_reward_score(reward_model, dataloader, accelerator)
    compute_reward_time = time.time() - start_time

    rewards = rewards[: len(generated_dataset)]
    generated_dataset = generated_dataset.add_column("rewards", rewards)

    # Save the dataset with rewards
    rewarded_dataset_path = script_args.save_rewarded_dataset_path
    #generated_dataset.save_to_disk(rewarded_dataset_path)
    #print(f"Saved dataset with rewards to {rewarded_dataset_path}")

    # Rejection sampling
    df = generated_dataset.to_pandas()
    df = df.groupby("prompt").agg({"response": lambda x: list(x), "rewards": lambda x: list(x)}).reset_index()
    start_time = time.time()
    df["accepted"], df["rewards"] = zip(*df.apply(
        lambda x: conduct_rejection_sampling(
            x["response"], 
            x["rewards"], 
            script_args.num_samples, 
            script_args.beta
        ),
        axis=1
    ))
    rejection_sampling_time = time.time() - start_time

    accepted_rewards = [reward for rewards in df["rewards"] for reward in rewards]

    # Ranking
    ranking_fn = tournament_ranking if "tournament" in script_args.ranking_method else first_round_ranking
    df["chosen"], df["rejected"] = zip(*df.apply(lambda x: ranking_fn(x["accepted"], x["rewards"]), axis=1))

    # Construct the final dataset
    df = df.filter(["prompt", "chosen", "rejected"])
    df = df.explode(["chosen", "rejected"])
    
    dataset = Dataset.from_pandas(df)
    
    # save the dataset for later finetuning with DPO
    #dataset.save_to_disk(script_args.save_dataset_path_tobeDPO)

    # Print out the times 
    print(f"BATCH_SAMPLING_TIME={generate_time}")
    print(f"REWARD_COMPUTE_TIME={compute_reward_time}")
    print(f"REJECTION_SAMPLING_TIME={rejection_sampling_time}")
