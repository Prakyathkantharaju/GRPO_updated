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
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
import requests  # Added for the new reward function


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

########################
# Helper functions
########################

def format_reward_func(completions, target, thinking_level, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt, tk_level in zip(completions, target, thinking_level):
        try:
            # Prepend synthetic <think> tag (as it is prefilled in the prompt)
            completion = "<think>" + completion        
            # Check if the overall format is correct using a regex
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL) 
            
            # If the format is incorrect, reward is 0.
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                think_text = match.group(1).strip()
                # Split the think_text into sentences.
                # This approach splits on punctuation marks (., !, or ?) that are followed by whitespace (or end-of-string).
                sentences = [s for s in re.split(r'(?<=[.!?])\s+', think_text) if s.strip()]
                sentence_count = len(sentences)
                
                # Check the sentence count against the provided thinking level.
                if tk_level == 1:
                    # If thinking level is 1, expect 1 or 2 sentences.
                    if sentence_count in (1, 2):
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    # For levels >= 2, require at least a number of sentences equal to the thinking level.
                    if sentence_count >=  tk_level - 2:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            rewards.append(0.0)
            continue
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           rewards.append(0.0)
           continue
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            rewards.append(1.0)
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
        else:
            rewards.append(0.0)
      except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards



def simple_eq_reward_func(completion, target, numbers, **kwargs):
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
            return 0.0
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            return 1.0
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
        else:
            return 0.0
    except Exception:
            # If evaluation fails, reward is 0
            return 0.0 


def simple_instruction_following_reward_func(completion, target, embeddings, **kwargs):
    """
    Calculates reward for instruction following tasks based on embedding differences.
    
    It computes the squared difference between the embedding of the generated completion 
    and the provided target embedding. In formula form, if:
      
      completion_embedding = E(completion)
      target_embedding = embeddings

    then the reward is given by:
      
      reward = sum((completion_embedding[i] - target_embedding[i])**2) for i in all dimensions

    Args:
        completion (str): Generated text from the model.
        target (str): Expected target text (not used in the direct computation here).
        embeddings (list[float]): Pre-computed target embedding.
        **kwargs: Additional keyword arguments.
      
    Returns:
        float: The computed squared difference between the computed embedding for the 
               completion and the target embedding.
    """
    def embed_text_api(text):
        """
        Obtains the embedding for the provided text using a POST request.
        """
        url = "http://127.0.0.1:8080/embed"
        response = requests.post(url, json={'inputs': text})
        response.raise_for_status()
        return response.json()

    try:
        # Get the embedding for the generated completion.
        completion_embedding = embed_text_api(completion)

        # If the API returns an embedding that is not a list or if the dimensions do not match, return 0.
        if not (isinstance(completion_embedding, list) and isinstance(embeddings, list)):
            return 0.0

        if len(completion_embedding) != len(embeddings):
            return 0.0

        # Calculate the squared difference (L2 norm squared).
        squared_diff = sum((ce - te) ** 2 for ce, te in zip(completion_embedding, embeddings))
        return squared_diff

    except Exception as e:
        # In case of any errors (network, format issues, etc.), return 0 reward.
        return 0.0

def equation_reward_func_2(completions, target, nums, source, embeddings, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers, source, embeddings in zip(completions, target, nums, source, embeddings):
        if source == "dataset_0":
            try:
                # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
                completion = "<think>" + completion
                # Check if the format is correct
                match = re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
                
                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue
                
                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builtins__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                    if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                        os.makedirs("completion_samples", exist_ok=True)
                        log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                        with open(log_file, "a") as f:
                            f.write(f"\n\n==============\n")
                            f.write(completion)
                else:
                    rewards.append(0.0)
            except Exception:
                    # If evaluation fails, reward is 0
                    rewards.append(0.0) 
        else:
            rewards.append(simple_instruction_following_reward_func(completion, gt, embeddings))
    return rewards

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
    dataset_0 = dataset_0.shuffle(seed=42).select(range(20000))
    dataset_1 = load_dataset('alpaca_dataset_with_embeddings', split='train')
    dataset_1 = dataset_1.shuffle(seed=42).select(range(5000))
    dataset_2 = load_dataset('teacher_dataset_with_embeddings', split='train')
    dataset_2 = dataset_2.shuffle(seed=42).select(range(5000))
    dataset_3 = load_dataset('roleplay_dataset_with_embeddings', split='train')
    dataset_3 = dataset_3.shuffle(seed=42).select(range(5000))

    def add_thinking_level(dataset):
        dataset = dataset.map(lambda x: {"thinking_level": random.randint(1, 10)})
        return dataset
    
    dataset_0 = add_thinking_level(dataset_0)
    dataset_1 = add_thinking_level(dataset_1)
    dataset_2 = add_thinking_level(dataset_2)
    dataset_3 = add_thinking_level(dataset_3)
    # merge the datasets but with an identity column to identify the source of the data
    dataset_0 = dataset_0.map(lambda x: {"source": "dataset_0"})
    dataset_1 = dataset_1.map(lambda x: {"source": "dataset_1"})
    dataset_2 = dataset_2.map(lambda x: {"source": "dataset_2"})
    dataset_3 = dataset_3.map(lambda x: {"source": "dataset_3"})


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
                "content": f"Using the numbers {numbers}, create an equation that equals {target}, thinking level is {thinking_level}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags. The magnitude of thinking is controlled by a thinking level from 1 - 10, where 1 is no thinking and 100 is large thinking time i.e if you are given level is 1 then you should only think for 1 or 2 sentences and if you given level is 10 then you can think for lot more sentence and perform a longer reasoning for your answer."
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }]
            return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target, "nums": numbers, "source": source, "thinking_level": thinking_level, "embeddings": None}

    def generater_r1_prompt_instruction_following(instruction, input_, response, thinking_level, source, embeddings):
        if source == "dataset_1":
            # Instruction following prompts for alpaca dataset
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
            return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": response, "nums": None, "source": source, "thinking_level": thinking_level, "embeddings": embeddings}
    

    dataset_0 = dataset_0.map(lambda x: generate_r1_prompt(x["nums"], x["target"], x["thinking_level"], x["source"]))
    dataset_1 = dataset_1.map(lambda x: generater_r1_prompt_instruction_following(x["instruction"], x["input"], x["response"], x["thinking_level"], x["source"], x["embeddings"]))
    dataset_2 = dataset_2.map(lambda x: generater_r1_prompt_instruction_following(x["instruction"], x["input"], x["response"], x["thinking_level"], x["source"], x["embeddings"]))
    dataset_3 = dataset_3.map(lambda x: generater_r1_prompt_instruction_following(x["instruction"], x["input"], x["response"], x["thinking_level"], x["source"], x["embeddings"]))

    # Split test and train and then combine
    dataset_0 = dataset_0.train_test_split(test_size=0.1)
    dataset_1 = dataset_1.train_test_split(test_size=0.1)
    dataset_2 = dataset_2.train_test_split(test_size=0.1)
    dataset_3 = dataset_3.train_test_split(test_size=0.1)

    # merge the datasets
    train_dataset = concatenate_datasets([dataset_0["train"], dataset_1["train"], dataset_2["train"], dataset_3["train"]])
    test_dataset = concatenate_datasets([dataset_0["test"], dataset_1["test"], dataset_2["test"], dataset_3["test"]])

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
      reward_funcs=[format_reward_func, equation_reward_func],
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