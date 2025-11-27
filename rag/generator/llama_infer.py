"""LLaMA inference wrapper"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


class LlamaGenerator:
    """Generate answers using LLaMA"""
    
    def __init__(self, model_name: str, max_new_tokens: int = 512, temperature: float = 0.7):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    
    def generate(self, prompt: str) -> str:
        """Generate response"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()
        return answer
