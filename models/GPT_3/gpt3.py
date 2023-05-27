import openai
from nltk import word_tokenize, sent_tokenize
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import tiktoken

from transformers import GPT2Tokenizer
class GPTInference():
    def __init__(self):
        openai.api_key = "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"
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_chatgpt_response(self, prompt):
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301",
                                       messages=[
                        {"role": "user", "content": prompt},
                        ], 
                        )
        return response['choices'][0]['message']['content']
    
    def make_summary_prompt(self, article, instruction = ''):
        prompt = f'Article: {article}\n{instruction}'
        return prompt
    
    def get_news_response(self, article, instruction):
#         tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

#         article_ids = tokenizer.encode(article)
#         instruction_ids = tokenizer.encode(instruction)
# #         article_token_limit = (4096 - len(instruction_ids)) - len(article_ids)
# #         article_ids = article_ids[:article_token_limit]
#         article = tokenizer.decode(article_ids)
        
        prompt = f'Article: {article}\n{instruction}'
        # print(prompt)
        response = self.get_chatgpt_response(prompt)
        return response