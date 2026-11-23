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
    output  = [-100 for i in range(len(labels))]
    for i in range(len(labels)) : 
        if i>=prompt_length:
            output[i]=labels[i]

    return output

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

# Step 16 - make_attention_mask
def make_attention_mask(padded_ids, pad_id):
    # TODO: return a same-shape 0/1 mask with 1 where token != pad_id else 0
    
    padded_ids = torch.tensor(padded_ids)
    mask= torch.where(padded_ids!=pad_id, 1,0)

    return mask.tolist()

# Step 17 - collate_lm_batch
def collate_lm_batch(batch, pad_id):
    # TODO: pad input_ids and labels, build attention mask, return dict of LongTensors
    
    # create separate list
    inputs_ids = [torch.tensor(example['input_ids']) for example in batch]
    labels = [torch.tensor(example['labels']) for example in batch]

    # apply the padding 
    inputs_ids = torch.nn.utils.rnn.pad_sequence(
        inputs_ids, 
        batch_first = True, 
        padding_value = pad_id
    )
    # create the attention mask according to the padding
    attention_mask = torch.where(inputs_ids!=pad_id,1,0)
    labels =torch.nn.utils.rnn.pad_sequence(
        labels, 
        batch_first=True, 
        padding_value=-100
    ) 
    #  return labels , attention_mask

    return {
        "labels" : labels , 
        "input_ids" : inputs_ids.to(dtype = torch.long), 
        "attention_mask" : attention_mask
    }

# Step 18 - iterate_minibatches
import random

def iterate_minibatches(examples, batch_size, seed=0):
    rng = random.Random(seed)

    shuffled = examples.copy()
    rng.shuffle(shuffled)

    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start:start + batch_size]

# Step 19 - train_val_split
def train_val_split(examples, val_ratio=0.2, seed=0):
    generator = torch.Generator().manual_seed(seed)

    idx = torch.randperm(
        len(examples), generator=generator
    ).tolist()

    # Shuffle examples
    shuffled = [examples[i] for i in idx]

    # Split
    n_train = len(examples) - int(val_ratio * len(examples))

    train = shuffled[:n_train]
    val = shuffled[n_train:]

    return train, val

# Step 20 - shift_logits_and_labels
def shift_logits_and_labels(logits, labels):
    # TODO: drop the last logit position and the first label position so token t predicts t+1
    
    return logits[:,:-1] , labels[:,1:]

# Step 21 - cross_entropy_loss
import torch
import torch.nn.functional as F

def cross_entropy_loss(shift_logits, shift_labels):
    """Mean next-token cross-entropy, ignoring label positions equal to -100."""
    # TODO: reduce (B, T-1, V) logits and (B, T-1) labels to a scalar loss tensor.
    # detect the position -100
    i,j = torch.where(shift_labels!=-100)
    selected_logits = shift_logits[i,j] # select the valid index 
    selected_labels  = shift_labels[i,j]
    selected_probs= torch.softmax(selected_logits, dim=-1)
    # select the right position 
    selected_probs = torch.gather(selected_probs, dim=-1, index =selected_labels.unsqueeze(-1)).squeeze(-1)

    return -torch.mean(torch.log(selected_probs+10e-12))

# Step 22 - adamw_update
import torch

def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Apply one in-place AdamW step to `param` using `grad` and persistent `state`."""
    # TODO: initialize state on first call, then update moments and apply the decoupled AdamW step
    
    if len(state)== 0: 
        #initiliaze the state 
        state["step"] = 0
        state["m"] = torch.zeros_like(param)
        state["v"] = torch.zeros_like(param)

    #update the momentum 
    state["step"]+=1
    state["m"].mul_(betas[0])
    state["m"].add_((1-betas[0])*grad)
    state["v"].mul_(betas[1])
    state["v"].add_((1-betas[1])*(grad**2))

    #remove the bias 
    m_bias = 1/(1-betas[0]**state['step'])
    v_bias = 1/(1-betas[1]**state["step"])
    m_hat = state["m"]*(m_bias)
    v_hat = state["v"]*(v_bias)

    # update the param 

    param.mul_((1-lr*weight_decay))
    param.add_(-(lr*m_hat)/(v_hat**0.5+eps))


    return param

# Step 23 - linear_warmup_schedule
def linear_warmup_schedule(step, warmup_steps):
    # TODO: return a linear warmup multiplier in [0, 1] given the current step and warmup window.
    # linear increasing from 0 to the warmup_step 
    # linear decreasing from that step to zero 
    if warmup_steps==0 : 
        return 1 

    return min(step, warmup_steps)/warmup_steps

# Step 24 - clip_grad_norm
def clip_grad_norm(grads, max_norm):
    # TODO: compute the global L2 norm of grads and rescale in place if it exceeds max_norm.

    grad_norm = torch.sum(torch.concat(grads)**2)**0.5

    if grad_norm>max_norm : 
        scale = max_norm/grad_norm.item()
    else : 
        scale = 1

    #update the grads
    for grad in grads : 
        grad.mul_(scale)

    return grad_norm

# Step 25 - accumulate_gradients
import torch

def accumulate_gradients(grad_list):
    """Average a list of equally-shaped gradient tensors across micro-batches."""
    # TODO: average a list of equally-shaped gradient tensors and return the mean tensor
    

    grads = torch.stack(grad_list, dim=-1)

    return torch.mean(grads, dim=-1)

