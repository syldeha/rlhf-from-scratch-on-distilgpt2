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

# Step 8 - top_p_filter
def top_p_filter(logits, p):
    # TODO: mask logits outside the smallest cumulative-probability nucleus of size p.
    logits = torch.tensor(logits)
    probats = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_index = torch.sort(probats, dim=-1, descending=True)

    #apply the cumsum 
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    position = (cum_probs<p).sum(dim=-1, keepdim=True)+1 # (batch_size, 1)

    # build the mask 
    mask = torch.arange(logits.shape[-1])[None, :]<position #(batch_size, vocab_size)
    mask = mask.to(dtype = torch.bool)
    sorted_logits = logits[sorted_index]
    sorted_logits = sorted_logits.masked_fill(~mask, -torch.inf).squeeze(0)


    output = torch.zeros_like(logits)

    output = torch.scatter(output, index = sorted_index, src = sorted_logits, dim=-1)


    return output

# Step 9 - build_synthetic_instruction_dataset
def build_synthetic_instruction_dataset():
    return [
        {
            "prompt": "Translate 'hello' to French.",
            "response": "Bonjour."
        },
        {
            "prompt": "What is 2 + 2?",
            "response": "2 + 2 equals 4."
        },
        {
            "prompt": "Name the capital of France.",
            "response": "The capital of France is Paris."
        },
        {
            "prompt": "Write a short greeting.",
            "response": "Hello! I hope you are having a great day."
        },
        {
            "prompt": "What color is the sky on a clear day?",
            "response": "The sky is typically blue on a clear day."
        }
    ]

# Step 10 - format_example
def format_example(example):
    # TODO: render {'prompt','response'} into one training string with role markers
    
    
    s= f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"

    return s

# Step 11 - apply_template
def apply_template(examples):
    # TODO: apply format_example to each item in examples and return the list of strings.
    output = [format_example(example) for example in examples]


    return output

# Step 12 - tokenize_example
def tokenize_example(tokenizer, text, max_length=64):
    # TODO: encode `text` with truncation at max_length, no padding, return list[int]
    
    return tokenizer.encode(
        text,
        truncation=True,
        max_length=max_length,
        padding=False
    )

# Step 13 - build_labels
def build_labels(input_ids):
    # TODO: return a fresh list equal to input_ids to serve as next-token labels
    
    
    return input_ids.copy()

# Step 14 - mask_prompt_labels
def mask_prompt_labels(labels, prompt_length):
    # TODO: replace the first prompt_length entries of labels with -100 and return the new list
    labels[:prompt_length] = -100

    return labels

# Step 15 - pad_batch
def pad_batch(sequences, pad_id):
    # TODO: right-pad a list of token id sequences to the longest length using pad_id
    
    max_len = 0
    for seq in sequences : 
        max_len = max(max_len, len(seq))
    sequences = [torch.tensor(seq) for seq in sequences]
    


    # apply the padding 

    sequences= torch.nn.utils.rnn.pad_sequence(
        sequences, 
        batch_first = True,
        padding_value = pad_id
    )

    return sequences.tolist()

