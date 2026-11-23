"""
RLHF from Scratch on DistilGPT2

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_distilgpt2_tokenizer
from transformers import AutoTokenizer
def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    # TODO: load and return the Hugging Face tokenizer for the given model name.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer

# Step 2 - load_distilgpt2_model
from transformers import AutoModelForCausalLM
def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    # TODO: load a causal LM by name and return it in eval mode
    model = AutoModelForCausalLM.from_pretrained(model_name)


    return model

# Step 3 - set_pad_token_to_eos
def set_pad_token_to_eos(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 4 - generate_and_decode
def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    # TODO: tokenize prompt, generate continuation greedily, decode and return as a string
    # if tokenizer.pad_token is None : 
    #     tokenizer.pad_token = tokenizer.eos_token

    tokenizer = set_pad_token_to_eos(tokenizer)
    inputs = tokenizer(
        prompt, return_tensors = 'pt'
    )
    inputs = {
        k : v.to(model.device)
        for k , v in inputs.items()
    }

    with torch.no_grad() : 

        output = model.generate(
        **inputs , max_new_tokens=max_new_tokens, 
        do_sample = False,
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(
        output[0] , skip_special_tokens=True
    )
    return response

# Step 5 - greedy_decode
import torch

def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    # TODO: return the token id with the largest logit as a Python int
    index = torch.argmax(logits, dim=-1)

    return index.item()

# Step 6 - sample_with_temperature
def sample_with_temperature(logits, temperature):
    # TODO: rescale logits by temperature, softmax, and sample one token id
    if temperature<=0 : 
        return torch.argmax(logits).item()
    else : 
        logits = logits/temperature

        # compute the softmax
        probs = torch.softmax(logits, dim=-1)

        return torch.multinomial(probs, num_samples=1).item()

# Step 7 - top_k_filter
def top_k_filter(logits, k):
    # TODO: keep the k largest entries of logits and set the rest to -inf.
    if k> logits.shape[-1] : 
        return logits
    topk_val , topk_index  = torch.topk(logits, k=k)

    mask = torch.zeros_like(logits)
    mask[topk_index] = 1
    mask = mask.to(dtype = torch.bool)
    logits = logits.masked_fill(~mask, -float("inf"))
    
    return logits

