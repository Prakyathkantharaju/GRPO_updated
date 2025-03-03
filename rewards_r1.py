import os
import random
import re
import requests
import torch

class RewardFunctions:
    """
    A class containing reward functions to evaluate model completions.
    Each method replicates a reward function and follows the same API
    as before, but without nested functions.
    """
    def __init__(self, gamma=0.5, beta=0.5, tokenizer=None):
        self.gamma = gamma
        self.beta = beta
        self.tokenizer = tokenizer
    def format_reward(self, completions, **kwargs):
        """
        Calculates a reward based on the formatting of the model completions.

        The expected format is:
          <think>...</think>
          <answer>...</answer>
        
        For each completion, a synthetic <think> tag is prepended, and its content is
        checked using a regular expression. The reward is decided based on the number 
        of sentences extracted from the <think> tags compared with a provided thinking level.

        Args:
            completions (list[str]): Generated outputs.
            target (list[str]): Expected answers (unused here).
            thinking_levels (list[int]): Thinking levels for each completion controlling the expected sentence count.
            **kwargs: Additional parameters (unused).

        Returns:
            list[float]: Reward scores (1.0 or 0.0) for each completion.
        """
        rewards = []
        for completion in completions:
            try:
                comp = "<think>" + completion  # Prepend synthetic tag
                # Regex to check correct format with <think> and <answer> tags.
                regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
                match = re.search(regex, comp, re.DOTALL)
                if match is None or len(match.groups()) != 2:
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)
                    # think_text = match.group(1).strip()
                    # # Split text into sentences based on punctuation.
                    # sentences = [s for s in re.split(r'(?<=[.!?])\s+', think_text) if s.strip()]
                    # sentence_count = len(sentences)
                    # if tk_level == 1:
                    #     # Expect 1 or 2 sentences for thinking level 1.
                    #     rewards.append(1.0 if sentence_count in (1, 2) else 0.0)
                    # else:
                    #     # For higher levels, require a minimum number of sentences.
                    #     rewards.append(1.0 if sentence_count >= tk_level - 2 else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards[0]

    def equation_reward(self, completions, target, nums, **kwargs):
        """
        Evaluates model completions for mathematical correctness.

        For each completion, it:
          1. Prepends a synthetic <think> tag.
          2. Extracts the equation from within <answer> tags.
          3. Checks if the equation uses all numbers exactly once.
          4. Uses a regex pattern to allow only the permitted characters.
          5. Evaluates the equation and compares it to the ground truth.

        Args:
            completions (list[str]): Generated outputs.
            target (list[str]): Expected numerical answers.
            nums (list[list[int]]): Available numbers for each completion.
            **kwargs: Additional parameters (unused).

        Returns:
            list[float]: Reward scores (1.0 for correct, 0.0 otherwise).
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                comp = "<think>" + completion
                match = re.search(r"<answer>(.*?)<\/answer>", comp)
                if match is None:
                    rewards.append(0.0)
                    continue
                equation = match.group(1).strip()
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                   rewards.append(0.0)
                   continue
                result = eval(equation, {"__builtins__": None}, {})
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                    if random.random() < 0.10:  # 10% chance to log successful samples
                        os.makedirs("completion_samples", exist_ok=True)
                        log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                        with open(log_file, "a") as f:
                            f.write(f"\n\n==============\n")
                            f.write(comp)
                else:
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    # def coupled_reward(self, completions, target, nums, **kwargs):
    #     rewards = []
    #     gamma = self.gamma
    #     beta = self.beta
    #     for completion, gt, numbers in zip(completions, target, nums):
    #         r_f = self.format_reward([completion], [gt], [numbers])
    #         r_e = self.simple_eq_reward(completion, gt, numbers)
    #         rewards.append(r_e + (max(r_e, 0) * r_f))
    #     return rewards


    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        # Ensure input_ids are of type Long
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
        ).logits  # (B, L, V)
        # print(logits.shape)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        # print(logits_to_keep)
        for logits_row, input_ids_row in zip(logits, input_ids[:, -logits_to_keep+1:]):
            # print(logits_row.shape, input_ids_row.shape)
            # print(logits_row.log_softmax(dim=-1).shape)
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
            
        return torch.stack(per_token_logps)


    def coupled_reward(self, model, completions, full_response, **kwargs):
        rewards = []
        gamma = self.gamma
        beta = self.beta
        for completion, target in zip(completions, full_response):
            r_f = self.format_reward([completion])
            r_e = self.dpo_reward(model, completion, target)
            rewards.append(r_e + (max(r_e, 0) * r_f))
        return rewards

    def dpo_reward(self, model, completion, full_response, **kwargs):
        chosen_encoding = self.tokenizer([full_response], return_tensors="pt")
        chosen_input_ids = chosen_encoding.input_ids
        chosen_attention_mask = chosen_encoding.attention_mask
        chosen_logits_to_keep = chosen_input_ids.shape[1]
        
        # Process rejected completion
        rejected_encoding = self.tokenizer([completion], return_tensors="pt")
        rejected_input_ids = rejected_encoding.input_ids
        rejected_attention_mask = rejected_encoding.attention_mask
        rejected_logits_to_keep = rejected_input_ids.shape[1]
        
        with torch.inference_mode():
            # Ensure all tensors are on the same device as the model
            device = model.device
            chosen_input_ids = chosen_input_ids.to(device)
            chosen_attention_mask = chosen_attention_mask.to(device).long()  # Ensure it's long type
            rejected_input_ids = rejected_input_ids.to(device)
            rejected_attention_mask = rejected_attention_mask.to(device).long()  # Ensure it's long type
            
            # Get log probs for chosen completion
            chosen_logprobs = self._get_per_token_logps(
                model, 
                chosen_input_ids, 
                chosen_attention_mask, 
                chosen_logits_to_keep
            )
            
            # Get log probs for rejected completion
            rejected_logprobs = self._get_per_token_logps(
                model, 
                rejected_input_ids, 
                rejected_attention_mask, 
                rejected_logits_to_keep
            )
        
        # Calculate reward difference (log prob normalized by length)
        chosen_reward = torch.sum(chosen_logprobs) / chosen_logprobs.shape[1]
        rejected_reward = torch.sum(rejected_logprobs) / rejected_logprobs.shape[1]
        
        # DPO reward: log(sigmoid(β * (r_chosen - r_rejected)))
        # Note: removed the negative sign from the loss formula
        reward_diff = - (chosen_reward - rejected_reward)
        return reward_diff

    def simple_eq_reward(self, completion, target, numbers, **kwargs):
        """
        Evaluates a single completion's correctness with a simple equation reward.

        Args:
            completion (str): Generated output.
            target (str or float): Expected answer.
            numbers (list[int]): Numbers that should be used exactly once.
            **kwargs: Additional parameters (unused).

        Returns:
            float: Reward score (1.0 for correct, 0.0 otherwise).
        """
        try:
            comp = "<think>" + completion
            match = re.search(r"<answer>(.*?)<\/answer>", comp)
            if match is None:
                return 0.0
            equation = match.group(1).strip()
            used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
            if sorted(used_numbers) != sorted(numbers):
                return 0.0
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                return 0.0
            result = eval(equation, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                if random.random() < 0.10:  # 10% chance to log successful samples
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(comp)
            return 1.0 if abs(float(result) - float(target)) < 1e-5 else 0.0
        except Exception:
            return 0.0

    def embed_text_api(self, text):
        """
        Obtains an embedding for the given text using a POST request.

        Args:
            text (str): The input text.

        Returns:
            list[float]: The embedding vector.
        """
        url = "http://127.0.0.1:8080/embed"
        response = requests.post(url, json={'inputs': text})
        response.raise_for_status()
        return response.json()

    def simple_instruction_following_reward(self, completion, target, embeddings, **kwargs):
        """
        Calculates reward for instruction-following tasks based on embedding differences.

        It computes the squared difference (L2 norm squared) between the embedding for the 
        generated completion and the provided target embedding.

        Args:
            completion (str): Generated text.
            target (str): Expected target text (unused in the calculation).
            embeddings (list[float]): Pre-computed target embedding.
            **kwargs: Additional parameters (unused).

        Returns:
            float: The squared L2 distance between the embeddings. Returns 0.0 on error.
        """
        try:
            completion_embedding = self.embed_text_api(completion)
            if not (isinstance(completion_embedding, list) and isinstance(embeddings, list)):
                return 0.0
            if len(completion_embedding) != len(embeddings):
                return 0.0
            squared_diff = sum((ce - te) ** 2 for ce, te in zip(completion_embedding, embeddings))
            return squared_diff
        except Exception:
            return 0.0

    def equation_reward_2(self, completions, target, response, nums, source, embedding, **kwargs):
        """
        Evaluates completions either based on mathematical correctness or on instruction following.

        If the source is "dataset_0", it performs a mathematical evaluation (same as equation_reward).
        For other sources, it uses the instruction following reward.

        Args:
            completions (list[str]): Generated outputs.
            target (list[str]): Expected answers.
            nums (list[list[int]]): Numbers that should be used.
            sources (list[str]): Identifier for the source of each completion.
            embeddings (list[Optional[list[float]]]): Target embeddings for non-dataset_0 completions.
            **kwargs: Additional parameters (unused).

        Returns:
            list[float]: Reward scores for each completion.
        """
        rewards = []
        for completion, gt, numbers, res,  src, embed in zip(completions, target, nums, response,  source, embedding):
            if src == "dataset_0":
                rewards.append(self.simple_eq_reward(completion, gt, numbers))
            else:
                rewards.append(self.simple_instruction_following_reward(completion, res, embed))
        return rewards 