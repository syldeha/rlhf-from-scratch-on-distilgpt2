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
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:] , skip_special_tokens=True
    )
    return response

