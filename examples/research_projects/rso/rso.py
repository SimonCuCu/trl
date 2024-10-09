import random
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

from accelerate import Accelerator, DistributedType
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedModel
)
#import model_training.models.reward_model

from trl.trainer.utils import conduct_rejection_sampling, compute_reward_score
#import deepspeed
#from deepspeed.utils.timer import SynchronizedWallClockTimer
#timers = SynchronizedWallClockTimer()


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

@dataclass
class ScriptArguments:
    reward_model_name_or_path: Optional[str] = field(default="output/checkpoint-20500", metadata={"help": "the model name"})
    mixed_precision: Optional[str] = field(default=None, metadata={"help": "the model dtype"})
    # data parameters
    dataset_name: Optional[str] = field(default="/scratch/bcjw/chuang8/trl_rso/sft_gen_dataset_v2", metadata={"help": "the generated dataset path"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the scoring batch size"})
    save_dataset_path: Optional[str] = field(default="sft_gen_dataset_ranked", metadata={"help": "the path for saving the dataset"})
    save_reward_dataset_path: Optional[str] = field(default="reward_dataset_test2", metadata={"help": "the path for saving the reward dataset"})
    save_rewarded_dataset_path: Optional[str] = field(default="rewarded_dataset_tobeRS", metadata={"help": "the path for saving the dataset with rewards"})

    # rejection sampling 
    num_samples: Optional[int] = field(default=8, metadata={"help": "the number of samples to keep after rejection sampling"})
    beta: Optional[int] = field(default=0.5, metadata={"help": "TO DO"})
    ranking_method: Optional[str] = field(default="first_round", metadata={"help": " or tournament TO DO"})
    #subset_size: Optional[int] = field(default=6000, metadata={"help": "the number of samples to use from the dataset for debugging"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False)


def first_round_ranking(responses: List[str], rewards: List[float]) -> Tuple[List[str], List[str]]:
    """Conducts first round ranking. Starts from n responses and construct n/2 pairs to be assigned
    to chosen or rejected based on there rewards.
    
    Args:
        responses: accecpted candidates from rejection sampling
        rewards: response rewards.
        
    Returns:
        chosen: chosen samples.
        rejected: rejected samples.
    """
    
    chosen = []
    rejected = []
    
    def pick(responses):
        selected = random.randrange(len(responses))
        return responses.pop(selected)
    
    responses = [(response, reward) for response, reward in zip(responses,rewards)]
    while responses:
        selected1 = pick(responses)
        selected2 = pick(responses)
        if selected1[1]>selected2[1]:
            chosen.append(selected1[0])
            rejected.append(selected2[0])
        else:
            chosen.append(selected2[0])
            rejected.append(selected1[0])
            
    return chosen, rejected


def tournament_ranking(responses: List[str], rewards: List[float]):
    """Conducts tournament ranking. Starts from n responses and construct n-1 pairs to be assigned
    to chosen or rejected based on there rewards.
    
    Args:
        responses: accecpted candidates from rejection sampling.
        rewards: response rewards.
        
    Returns:
        chosen: chosen samples.
        rejected: rejected samples.
    """
    sorted_responses = [response for _, response in sorted(zip(rewards, responses), reverse=True)]
    
    chosen = [sorted_responses[i] for i in range(0, len(responses), 2)]
    rejected =[sorted_responses[i] for i in range(1, len(responses), 2)]
    
    return chosen, rejected


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    
    if script_args.num_samples%2!=0:
        warnings.warn(
            "Creating pairs requires an even number for num_samples."
            f"Setting num_samples to {script_args.num_samples+1} instead of {script_args.num_samples}"
        )
        script_args.num_samples += 1

    accelerator = Accelerator()

    # load reward model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(script_args.reward_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model_name_or_path)

    # Set padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    #print(repr(tokenizer.pad_token))
    model.config.pad_token_id = tokenizer.pad_token_id

    # load and preprocess the dataset
    dataset = load_from_disk(script_args.dataset_name)
    #dataset = dataset.select(range(min(len(dataset), script_args.subset_size)))


    if script_args.sanity_check:
        dataset = dataset.dataset(range(min(len(dataset), 500)))

    def tokenize_fn(samples):
        # create the text column first
        text = [prompt + " " + response for prompt, response in zip(samples["prompt"], samples["response"])]
        model_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        return {
            **model_inputs,
        }

    reward_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=list(dataset.features))
    
    # Save the preprocessed dataset to compute reward
    reward_dataset.save_to_disk(script_args.save_reward_dataset_path)
    print(f"Saved reward dataset to {script_args.save_reward_dataset_path}")

    data_collator = DataCollatorWithPadding(tokenizer)

    dataloader = DataLoader(reward_dataset, batch_size=script_args.batch_size, shuffle=False, collate_fn=data_collator)

    model, dataloader = accelerator.prepare(model, dataloader)
    
    #compute reward
    start_time=time.time()
    rewards = compute_reward_score(model, dataloader, accelerator)
    computeR_time = time.time() - start_time
    
    rewards = rewards[: len(dataset)]

    dataset = dataset.add_column("rewards", rewards)

    # Save the dataset with rewards
    rewarded_dataset_path = script_args.save_rewarded_dataset_path
    dataset.save_to_disk(rewarded_dataset_path)
    print(f"Saved dataset with rewards to {rewarded_dataset_path}")

    
    # perform rejection sampling
    df = dataset.to_pandas()
    df = df.groupby("prompt").agg({"response":lambda x: list(x), "rewards":lambda x: list(x)}).reset_index()
    
    # conduct rejected sampling algorithm as in https://arxiv.org/pdf/2309.06657.pdf
    start_time=time.time()
    df["accepted"], df["rewards"] = zip(*df.apply(
            lambda x: conduct_rejection_sampling(
                x["response"], 
                x["rewards"], 
                script_args.num_samples, 
                script_args.beta
            ),
            axis=1
        )
    )
    RS_time = time.time() - start_time
    

    # perform ranking
    
    ranking_fn = tournament_ranking if "tournament" in script_args.ranking_method else first_round_ranking
    start_time=time.time()
    df["chosen"], df["rejected"] = zip(*df.apply(lambda x: ranking_fn(x["accepted"], x["rewards"]), axis=1))
    rk_time = time.time() - start_time
    df = df.filter(["prompt", "chosen", "rejected"])
    df = df.explode(["chosen", "rejected"])
    
    dataset = Dataset.from_pandas(df)
    
    # save the dataset for later finetuning with DPO
    dataset.save_to_disk(script_args.save_dataset_path)
    print(f"Time taken to compute rewards: {computeR_time:.6f} seconds")
    print(f"Time taken to RS: {RS_time:.6f} seconds")
    print(f"Time taken to ranking: {rk_time:.6f} seconds")
