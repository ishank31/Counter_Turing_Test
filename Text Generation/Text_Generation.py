# pip install -q transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# change  model name here to get responses from different models
checkpoint = ""

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")


df = pd.read_csv('') # Read dataset  file

def get_response(headline):
  prompt = "Consider the given headline and write a news article for it in Hindi: " #This prefix changes for some models
  prompt += headline
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  input_ids=input_ids.to('cuda')
  generation_output = model.generate(input_ids=input_ids, max_new_tokens=250, pad_token_id=tokenizer.eos_token_id)
  return tokenizer.decode(generation_output[0])

# Define your range
start_index = # Enter start index
end_index = # Enter end index

# Create a tqdm progress bar
with tqdm(total=end_index - start_index + 1, desc="API Requests") as pbar:
    for index in range(start_index, end_index + 1):
        row = df.loc[index]
        cleaned_text = row['Link Text']  # Assuming 'cleaned_text' contains the text for API requests

        
        response = get_response(cleaned_text)

        # Update the DataFrame with the response
        df.at[index, 'responses'] = response

        # Update the tqdm progress bar
        pbar.update(1)

        # Sleep if needed to respect rate limits
        time.sleep(2)  # Example sleep duration

# Save file
df.to_csv('',index = False)

