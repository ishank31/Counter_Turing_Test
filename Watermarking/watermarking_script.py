from watermarking import Watermarker

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from indicnlp.tokenize import indic_tokenize
from bert_score import score

# import Levenshtein as lev
from Levenshtein import distance
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="start index", type=int, required=True)
parser.add_argument("--end", help="end index", type=int, required=True)
args = parser.parse_args()

watermarker = Watermarker(model="google/gemma-1.1-7b-it")

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
    input_text = "Revise this in Hindi only with your best efforts. " + input_text
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=500)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

df = pd.read_csv('') #Enter file path


result_df = pd.DataFrame(columns=['prompt', 'Watermarked_text','Paraphrased_text', 'pVal_after_paraphrasing', 'percent', 'edit_dist', 'bleu_score', 'bert_score'])


start_index = args.start
end_index = min(args.end, len(df) - 1)
# Process articles in the specified range
with tqdm(total=end_index - start_index + 1, desc="Progress") as pbar:
    for index in range(start_index, end_index + 1):
        article = df.at[index, 'Watermarked_text']

        
        text = watermarker.generateText('Write a Hindi news article for the headline. '+ article, 
						 		 m=500,                 # Number of tokens to generate
						 		 n=200,                 # Numbers of tokens to watermark
						 		 key=42,
						 		 seed=42,
						 		 verbose=False)

        # Update the DataFrame with the response
        result_df.at[index, 'Headline'] = article
        result_df.at[index, 'Watermarked_text'] = text

        # print("text done")
        #pVal_before = watermarker.detectWatermark(text,
							 n=200,
							 key=42,
							 verbose=False)

        #result_df.at[index, 'pVal_before_paraphrasing'] = pVal_before
        # print("pval before done")

        paraphrasedText = rewrite(text)
        result_df.at[index, 'Paraphrased_text'] = paraphrasedText
        # print("paraphrasing done")

         pVal_after = watermarker.detectWatermark(article,\
		  					 n=250,\
		  					 key=42,\
		  					 verbose=False)
        
        # print("pval after done")
        result_df.at[index, 'pVal_after_paraphrasing'] = pVal_after
        text_tokens = indic_tokenize.trivial_tokenize(text)
        paraphrased_tokens = indic_tokenize.trivial_tokenize(paraphrasedText)
        result_df.at[index, 'percent'] = df.at[index, 'percent']
        result_df.at[index, 'Watermarked_text'] = text
        result_df.at[index, 'Paraphrased_text'] = paraphrasedText
        result_df.at[index, 'pVal_after_paraphrasing'] = df.at[index, 'pVal_after_paraphrasing']
        
        P, R, F1 = score([text], [paraphrasedText], lang='hi')
        result_df.at[index, 'bert_score'] = F1[0]
        result_df.at[index, 'edit_dist'] = distance(text, paraphrasedText)
        result_df.at[index, 'bleu_score'] = bleu_score = sentence_bleu([text_tokens], paraphrased_tokens, smoothing_function=SmoothingFunction().method1)

        # Update progress bar
        pbar.update(1)
                

        # Save periodically
        if index % 10 == 0 or index == end_index:
            output_file_name = f''                                  # Enter output file name
            result_df.to_csv(output_file_name, index=False)
 
output_file_name = f''                      # Enter output file name
result_df.to_csv(output_file_name, index=False)
