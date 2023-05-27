from transformers import T5Tokenizer, T5ForConditionalGeneration


device = 'cuda'
class FlanT5Inference:
    
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", cache_dir="/scratch/ramprasad.sa/huggingface_models")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", cache_dir="/scratch/ramprasad.sa/huggingface_models")
        self.model.to(device)
#         self.tokenizer.to('cuda')

    def get_response(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        response = self.model.generate(input_ids, max_length = 1024)
        return self.tokenizer.decode(response[0], skip_special_tokens = True)

    
    def get_news_response(self,  article, instruction = '', token_limit = 4096):
        article_ids = self.tokenizer(article).input_ids
        instruction_ids = self.tokenizer(instruction).input_ids 
        article_token_limit = (4096 - len(instruction_ids)) - len(article_ids)
        
        article_ids = article_ids[:article_token_limit]
        article = self.tokenizer.decode(article_ids, skip_special_tokens = True )
        prompt = f'Article: {article}\n{instruction}'
        # print(prompt)
        
        response = self.get_response(prompt)
        # print(response)
        return response