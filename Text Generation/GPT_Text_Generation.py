import openai
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import time
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import argparse
import asyncio

#Enter your API key
openai.api_key = ""

#Get the two arguments from the command line
parser = argparse.ArgumentParser()
#Get --start and --end arguments from the command line
parser.add_argument("--start", help="start index", type=int)
parser.add_argument("--end", help="end index", type=int)
args = parser.parse_args()


# Change model name to GPT4 to get responses from GPT4
@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
async def turbo(prompt):
  # prompt += '. Expand this headline into a Hindi article'
  CONTEXT = "You are an assitant skilled in writing Hindi news articles."
  prompt = CONTEXT + "\n" + prompt + ". Expand this headline into a Hindi article." + "\n"
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      # {"role": "system", "content": "You are an assitant skilled in writing Hindi news articles."},
      {"role": "user", "content": prompt}
  ],
  
  )

  return (completion.choices[0].message.content)


df = pd.read_csv('') # Read dataset file

start_index = args.start
end_index = args.end
output_file = 'GPT3_5_response_'+ str(start_index) + '_' + str(end_index) + '.csv'

async def get_responses(df, start_index, end_index):
    # Create a tqdm progress bar
    with tqdm(total=end_index - start_index + 1, desc="API Requests") as pbar:
        for index in range(start_index, end_index + 1):
            try:
                row = df.loc[index]
                cleaned_text = row['Link Text']  # Assuming 'cleaned_text' contains the text for API requests

                
                response = await turbo(cleaned_text)

                # Update the DataFrame with the response
                df.at[index, 'GPT3.5_responses'] = response

                # Update the tqdm progress bar
                pbar.update(1)

                # Sleep if needed to respect rate limits
                time.sleep(1)  # Example sleep duration

            except Exception as e:
                print(f"Error at index {index}: {e}")
                df.at[index, 'GPT3.5_responses'] = "Error"

            finally:
                df.to_csv(output_file, index=False)


    return df

df = asyncio.run(get_responses(df, start_index, end_index))

df.to_csv(output_file, index=False)
