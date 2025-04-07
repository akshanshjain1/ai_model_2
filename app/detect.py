# app/detect.py
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math

model_name = "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model.eval()

def calculate_perplexity(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return math.exp(loss.item())
