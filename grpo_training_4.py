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
    dataset_id_or_path: str = "equation_dataset.json"  # Default to local dataset file
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
    # Import the create_dataset function from create_dataset_2.py
    from create_dataset_2 import create_dataset
    
    # Create or load the dataset
    logger.info(f"Creating/loading dataset from {script_args.dataset_id_or_path}")
    if os.path.exists(script_args.dataset_id_or_path):
        # Load from existing JSON file
        with open(script_args.dataset_id_or_path, 'r') as f:
            import json
            processed_data = json.load(f)
        from datasets import Dataset
        dataset_0 = Dataset.from_list(processed_data)
    else:
        # Create new dataset
        dataset_0, _ = create_dataset(save_path=script_args.dataset_id_or_path)
    
    dataset_0 = dataset_0.shuffle(seed=42)
    dataset_0 = dataset_0.select(range(100))

    #####################
    # Prepare and format dataset
    #####################
    
    # Apply tokenizer to the prompt field
    def tokenize_prompt(example):
        # Apply chat template to the prompt messages
        formatted_prompt = tokenizer.apply_chat_template(
            example["prompt"], 
            tokenize=False, 
            continue_final_message=True
        )
        return {"formatted_prompt": formatted_prompt}
    
    # Apply tokenization to the dataset
    dataset_0 = dataset_0.map(tokenize_prompt)
    
    # Rename the formatted_prompt field to prompt for the trainer
    dataset_0 = dataset_0.rename_column("formatted_prompt", "prompt")
    
    # Make sure thinking_level is available for the reward functions
    def prepare_for_reward(example):
        # Ensure thinking_level is an integer
        if isinstance(example["thinking_level"], str):
            example["thinking_level"] = int(example["thinking_level"])
        return example
    
    dataset_0 = dataset_0.map(prepare_for_reward)
    
    # Split test and train
    dataset_0 = dataset_0.train_test_split(test_size=0.1)
    
    train_dataset = dataset_0["train"]
    test_dataset = dataset_0["test"]
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)

    #########################
    # Instantiate GPRO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[rewards_functions.coupled_reward, rewards_functions.format_reward],
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
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "math"]})
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