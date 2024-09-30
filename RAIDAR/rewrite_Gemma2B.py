import os
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from indicnlp.tokenize import sentence_tokenize
import time



import argparse
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="start index", type=int, required=True)
parser.add_argument("--end", help="end index", type=int, required=True)
args = parser.parse_args()

# Model and tokenizer initialization
path = ""
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it", token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",
    device_map="auto",
    torch_dtype=torch.float16,
    revision="float16",
    token=os.environ['HF_TOKEN'],
    cache_dir=path
)



def rewrite(input_text):
    input_text = "Revise this in Hindi only with your best efforts. " + input_text   # Change your instruction as needed
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=500)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load dataset
df = pd.read_csv('') 

articles = list(df['text'])

# Shorten articles to first four sentences
short_articles = []
for article in tqdm(articles, desc="Tokenizing Articles"):
    sentences = sentence_tokenize.sentence_split(article, lang='hi')
    first_four_sentences = sentences[:4]
    sentence = " ".join(sentences)
    short_articles.append(sentence)

# Get start and end index
start_index = args.start
end_index = min(args.end, len(short_articles) - 1)

# Initialize DataFrame to store results
result_df = pd.DataFrame(columns=['input', 'Paraphrased_text'])

batch_size = 10
# Process articles in the specified range
with tqdm(total=end_index - start_index + 1, desc="API Requests") as pbar:
    for index in range(start_index, end_index + 1):
        article = short_articles[index]
        #print(f"Before processing article {index}: Current CUDA memory usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        # Make API request using the rewrite function
        response = rewrite(article)

        # Update the DataFrame with the response
        result_df.at[index, 'input'] = article
        result_df.at[index, 'Paraphrased_text'] = response

        # Update progress bar
        pbar.update(1)
        #print(f"After processing article {index}: Maximum CUDA memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        # Clear GPU memory and Python garbage collection
        

        # Save periodically
        if index % 10 == 0 or index == end_index:
            output_file_name = f''                           # Enter output file name
            result_df.to_csv(output_file_name, index=False)

# Final save
output_file_name = f''                                    # Enter output file name
result_df.to_csv(output_file_name, index=False)
