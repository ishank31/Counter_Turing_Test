import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Literal
import os

from watermark.demo.generate import generate_shift
from watermark.demo.detect import permutation_test
os.environ["HF_TOKEN"] = "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" # For Gemma 7B
# os.environ["HF_TOKEN"] = "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" # For Llama-2 7B
class Watermarker:
	def __init__(self, model : Literal["facebook/opt-1.3b"] = "facebook/opt-1.3b") -> None:
		"""
		Parameters:
			model : a HuggingFace model id of the model to generate from
		"""
		print("loading model!")
		self.tokenizer = AutoTokenizer.from_pretrained(model, token=os.environ['HF_TOKEN'])
		self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto",
    													torch_dtype=torch.float16,
    													token=os.environ['HF_TOKEN'],
    													cache_dir = "")
		
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		print("Watermarker initialized.")

	def generateText(self, prompt : str, m : int, n : int, key : int, seed : int = 0, verbose : bool = False) -> str:
		"""
		Generates watermarked text with the given prompt and watermarking key.

		Parameters:
			prompt : an optional prompt for generation
			m : the requested length of the generated text
			n : the length of the watermark sequence
			key : a key for generating the random watermark sequence
			seed : a seed for the random number generator
			verbose : print information or not?

		Returns:
			watermarked_text : the watermarked text
		"""
		torch.manual_seed(seed)
		
		tokens = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=2048)

		watermarked_tokens = generate_shift(self.model, tokens, len(self.tokenizer), n, m, key)[0]
		watermarked_text = self.tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

		if verbose:
			print(watermarked_text)

		return watermarked_text
	
	def detectWatermark(self, watermarkedText : str, n : int, key : int, verbose : bool = False) -> float:
		"""
		Detects the presence of a watermark in a given text document.

		Parameters:
			watermarkedText : the watermarked text
			n : the length of the watermark sequence
			key : a key for generating the random watermark sequence
			verbose : print information or not?
		"""
		tokens = self.tokenizer.encode(watermarkedText, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]

		startTime = time.time()
		pVal = permutation_test(tokens, key, n, len(tokens), len(self.tokenizer))

		if verbose:
			print('p-value: ', pVal)
			print(f'(elapsed time: {time.time()-startTime}s)')

		return pVal
