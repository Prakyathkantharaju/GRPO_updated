import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets
import datasets
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
import requests  # Added for the new reward function
from rewards_r1 import RewardFunctions

from datasets import Value, Sequence


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

rewards_functions = RewardFunctions(gamma=0.5, beta=0.5)



def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    dataset_0 = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    dataset_0 = dataset_0.shuffle(seed=42)
    # dataset_1 = datasets.load_from_disk('alpaca_dataset_with_embeddings')
    # dataset_1 = dataset_1.shuffle(seed=42).select(range(1000))
    # dataset_2 = datasets.load_from_disk('teacher_dataset_with_embeddings')
    # dataset_2 = dataset_2.shuffle(seed=42).select(range(1000))
    # dataset_3 = datasets.load_from_disk('roleplay_dataset_with_embeddings')
    # dataset_3 = dataset_3.shuffle(seed=42).select(range(1000))

    def add_thinking_level(dataset):
        dataset = dataset.map(lambda x: {"thinking_level": random.randint(1, 10)})
        return dataset
    
    dataset_0 = add_thinking_level(dataset_0)
    # dataset_1 = add_thinking_level(dataset_1)
    # dataset_2 = add_thinking_level(dataset_2)
    # dataset_3 = add_thinking_level(dataset_3)
    # merge the datasets but with an identity column to identify the source of the data
    dataset_0 = dataset_0.map(lambda x: {"source": "dataset_0"})
    # dataset_1 = dataset_1.map(lambda x: {"source": "dataset_1"})
    # dataset_2 = dataset_2.map(lambda x: {"source": "dataset_2"})
    # dataset_3 = dataset_3.map(lambda x: {"source": "dataset_3"})


    #####################
    # Prepare and format dataset
    #####################

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(numbers, target, thinking_level, source):
        if source == "dataset_0":
            r1_prefix = [{
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
            },
            { 
                "role": "user",
                "content": f"Using the numbers {numbers}, create an equation that equals {target}, thinking level is {thinking_level}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags. The magnitude of thinking is controlled by a thinking level from 1 - 10, where 1 is no thinking and 10 is large thinking time i.e if you are given level is 1 then you should only think for 1 or 2 sentences and if you given level is 10 then you can think for a lot more sentences and perform a longer reasoning for your answer."
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }]
            return {
                "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
                "target": target,
                'response': '',
                "nums": numbers,
                "source": source,
                "thinking_level": thinking_level,
                "embedding": [0.0] * 1024  # Consistent empty embedding
            }

    def generater_r1_prompt_instruction_following(instruction, input_, response, thinking_level, source, embeddings):
        if source in ["dataset_1", "dataset_2", "dataset_3"]:
            r1_prefix = [{
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
            },
            {
                "role": "user",
                "content": f"Here is the following instruction you need to follow {instruction}. Here is the user input to the instruction {input_}. You need to step by step think about the user instruction and input and produce the answer, you can perform thinking using the <think> </think> tags. And need to return the final answer in <answer> </answer> tags. The level of thinking specified by the used is {thinking_level}. i.e you need thinkin only for {thinking_level} of sentence in your thinking stage. "
            },
            {
                "role": "assistant",
                "content": "Let me think about the instruction and input. \n<think>"
            }]
            # print(response)
            return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
            "target": 0, 
            'response': response, 
            "nums": [0,0,0], 
            "source": source, 
            "thinking_level": thinking_level, 
            "embedding": embeddings if isinstance(embeddings, list) else embeddings.tolist()}
    

    dataset_0 = dataset_0.map(lambda x: generate_r1_prompt(x["nums"], x["target"], x["thinking_level"], x["source"]))
    # dataset_1 = dataset_1.map(lambda x: generater_r1_prompt_instruction_following(x["instruction"], x["input"], x["response"], x["thinking_level"], x["source"], x["embedding"]))
    # dataset_2 = dataset_2.map(lambda x: generater_r1_prompt_instruction_following(x["instruction"], x["input"], x["response"], x["thinking_level"], x["source"], x["embedding"]))
    # dataset_3 = dataset_3.map(lambda x: generater_r1_prompt_instruction_following(x["instruction"], x["input"], x["response"], x["thinking_level"], x["source"], x["embedding"]))

    # Split test and train and then combine
    dataset_0 = dataset_0.train_test_split(test_size=0.1)
    # dataset_1 = dataset_1.train_test_split(test_size=0.1)
    # dataset_2 = dataset_2.train_test_split(test_size=0.1)
    # dataset_3 = dataset_3.train_test_split(test_size=0.1)

    # Before concatenating, ensure all datasets have the same schema
    def standardize_features(dataset):
        features = datasets.Features({
            'instruction': datasets.Value('string'),
            'input': datasets.Value('string'),
            'response': datasets.Value('string'),
            'embedding': datasets.Sequence(datasets.Value('float64'), length=-1),
            'thinking_level': datasets.Value('int64'),
            'source': datasets.Value('string'),
            'prompt': datasets.Value('string'),
            'target': datasets.Value('string'),
            'nums': datasets.Sequence(datasets.Value('int64'), length=-1)
        })
        
        def convert_row(x):
            return {
                'instruction': str(x['instruction']) if x.get('instruction') is not None else '',
                'input': str(x['input']) if x.get('input') is not None else '',
                'response': str(x['response']) if x.get('response') is not None else '',
                'embedding': x['embedding'] if isinstance(x['embedding'], list) else x['embedding'].tolist(),
                'thinking_level': int(x['thinking_level']) if x.get('thinking_level') is not None else 0,
                'source': str(x['source']) if x.get('source') is not None else '',
                'prompt': str(x['prompt']) if x.get('prompt') is not None else '',
                'target': str(x['target']) if x.get('target') is not None else '',
                'nums': x['nums'] if isinstance(x['nums'], list) else [0, 0, 0]
            }
        
        # First map the conversion
        converted = dataset.map(convert_row)
        # Then cast to ensure consistent features
        return converted.cast(features)

    # Apply standardization before concatenation
    dataset_0["train"] = standardize_features(dataset_0["train"])
    # dataset_1["train"] = standardize_features(dataset_1["train"])
    # dataset_2["train"] = standardize_features(dataset_2["train"])
    # dataset_3["train"] = standardize_features(dataset_3["train"])

    dataset_0["test"] = standardize_features(dataset_0["test"])
    # dataset_1["test"] = standardize_features(dataset_1["test"])
    # dataset_2["test"] = standardize_features(dataset_2["test"])
    # dataset_3["test"] = standardize_features(dataset_3["test"])

    # # Print dataset information before concatenation
    # print("\nDataset Information Before Concatenation:")
    # print("\nDataset 0:")
    # print(f"Train shape: {len(dataset_0['train'])} samples")
    # print(f"Test shape: {len(dataset_0['test'])} samples") 
    # print("Features:", list(dataset_0['train'].features.keys()))
    # for key in dataset_0['train'].features.keys():
    #     print(f"Shape of {key}:", dataset_0['train'][key].shape if hasattr(dataset_0['train'][key], 'shape') else len(dataset_0['train'][key]))

    # print("\nDataset 1:") 
    # print(f"Train shape: {len(dataset_1['train'])} samples")
    # print(f"Test shape: {len(dataset_1['test'])} samples")
    # print("Features:", list(dataset_1['train'].features.keys()))
    # for key in dataset_1['train'].features.keys():
    #     print(f"Shape of {key}:", dataset_1['train'][key].shape if hasattr(dataset_1['train'][key], 'shape') else len(dataset_1['train'][key]))

    # print("\nDataset 2:")
    # print(f"Train shape: {len(dataset_2['train'])} samples") 
    # print(f"Test shape: {len(dataset_2['test'])} samples")
    # print("Features:", list(dataset_2['train'].features.keys()))
    # for key in dataset_2['train'].features.keys():
    #     print(f"Shape of {key}:", dataset_2['train'][key].shape if hasattr(dataset_2['train'][key], 'shape') else len(dataset_2['train'][key]))

    # print("\nDataset 3:")
    # print(f"Train shape: {len(dataset_3['train'])} samples")
    # print(f"Test shape: {len(dataset_3['test'])} samples")
    # print("Features:", list(dataset_3['train'].features.keys()))
    # for key in dataset_3['train'].features.keys():
    #     print(f"Shape of {key}:", dataset_3['train'][key].shape if hasattr(dataset_3['train'][key], 'shape') else len(dataset_3['train'][key]))

    # Now concatenate the datasets
    train_dataset = concatenate_datasets([dataset_0["train"]])
    test_dataset = concatenate_datasets([dataset_0["test"]])
    # Shuffle the train dataset
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)

    # convert our dataset to the r1 prompt
    # dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"], x["thinking_level"],)

    # split the dataset into train and test
    # train_test_split = dataset.train_test_split(test_size=0.1)

    # train_dataset = train_test_split["train"]
    # test_dataset = train_test_split["test"]

    #########################
    # Instantiate GPRO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[rewards_functions.coupled_reward],
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()