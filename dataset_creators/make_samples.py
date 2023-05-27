from transformers import T5Tokenizer, T5ForConditionalGeneration
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import dataset_creators.config as config
import pandas as pd
import os
import uuid
import argparse
from tqdm import tqdm




def check_prompt_token_limits(article, instructions, tokenizer, token_limit):
    counter = 0
    for key, instruction in instructions.items():
        prompt = f'Article: {article}\n{instruction}'
        prompt_len = len(tokenizer.encode(prompt))
        if prompt_len < token_limit:
            counter += 1 
    return counter 


def get_shortlisted_data(articles, reference_summaries, ids = [], token_limit = 4096, dataset = 'news'):
    
    flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    flan_instructions = config.instructions[f'{dataset}_flant5']
    
    gpt_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    gpt_instructions = config.instructions[f'{dataset}_gpt3']
    
    shortlisted_articles = []
    shortlisted_reference_summaries = []
    
    shortlisted_ids = []
    
    for idx, article in enumerate(tqdm(articles)):
        add_article = 0 
        
        flan_counter = check_prompt_token_limits(article, flan_instructions, flan_tokenizer, token_limit)
        add_article += flan_counter
        
        gpt_counter = check_prompt_token_limits(article, gpt_instructions, gpt_tokenizer, token_limit)
        add_article += gpt_counter
        

        if add_article == 4:
            shortlisted_articles.append(article)
            shortlisted_reference_summaries.append(reference_summaries[idx])
            
            if not ids:
                article_id = str(uuid.uuid4())
            else:
                article_id = ids[idx]
            
            shortlisted_ids.append(article_id)
            
    return shortlisted_articles, shortlisted_reference_summaries, shortlisted_ids




def make_sample_news(data_path, token_limit = 4096):
    cnndm_dataset = load_dataset("ccdv/cnn_dailymail", '3.0.0', split = 'test', cache_dir = '/scratch/ramprasad.sa/huggingface_datasets')
    xsum_dataset = load_dataset("xsum", split = 'test', cache_dir = '/scratch/ramprasad.sa/huggingface_datasets')
    
    
    
    shortlisted_data = {'article': [], 'reference_summary': [], 'id': [], 'origin': []}
    
    
    cnndm_articles = cnndm_dataset['article']
    cnndm_reference_summaries = cnndm_dataset['highlights']
    cnndm_ids = cnndm_dataset['id']
    
    shortlisted_articles, shortlisted_reference_summaries, shortlisted_ids = get_shortlisted_data(cnndm_articles, cnndm_reference_summaries, cnndm_ids)
    shortlisted_data['article'] += shortlisted_articles
    shortlisted_data['reference_summary'] += shortlisted_reference_summaries
    shortlisted_data['id'] += shortlisted_ids
    shortlisted_data['origin'] += ['cnndm'] * len(shortlisted_ids)
    
    xsum_articles = xsum_dataset['document']
    xsum_reference_summaries = xsum_dataset['summary']
    xsum_ids = xsum_dataset['id']
    
    shortlisted_articles, shortlisted_reference_summaries, shortlisted_ids = get_shortlisted_data(xsum_articles, xsum_reference_summaries, xsum_ids)
    shortlisted_data['article'] += shortlisted_articles
    shortlisted_data['reference_summary'] += shortlisted_reference_summaries
    shortlisted_data['id'] += shortlisted_ids
    shortlisted_data['origin'] += ['xsum'] * len(shortlisted_ids)
    
    isExist = os.path.exists(data_path)
    if not isExist:
        os.makedirs(data_path)
    
    df = pd.DataFrame(shortlisted_data)
    df.to_csv(f'{data_path}/test_sample.csv')
    return df


def make_sample_pubmed(data_path, token_limit = 4096):
    dataset = load_dataset("ccdv/pubmed-summarization", split="test", cache_dir = '/scratch/ramprasad.sa/huggingface_datasets')
    articles = dataset['article']
    reference_summaries = dataset['abstract']
    
    
    
    shortlisted_data = {'article': [], 'reference_summary': [], 'id': [], 'origin': []}
    
    shortlisted_articles, shortlisted_reference_summaries, shortlisted_ids = get_shortlisted_data(articles, reference_summaries, dataset = 'pubmed')
    shortlisted_data['article'] += shortlisted_articles
    shortlisted_data['reference_summary'] += shortlisted_reference_summaries
    shortlisted_data['id'] += shortlisted_ids
    shortlisted_data['origin'] += ['pubmed'] * len(shortlisted_ids)
            
    isExist = os.path.exists(data_path)
    if not isExist:
        os.makedirs(data_path)
    
    df = pd.DataFrame(shortlisted_data)
    df.to_csv(f'{data_path}/test_sample.csv')
    return df

def make_sample_chemsum(data_path, token_limit = 4096):
    dataset = load_dataset("griffin/ChemSum", split = 'test', cache_dir = '/scratch/ramprasad.sa/huggingface_datasets')
    articles = dataset['sections']
    articles = [preprocess_html_tags(each) for each in articles]
    reference_summaries = dataset['abstract']
    ids = dataset['uuid']
    
    
    shortlisted_data = {'article': [], 'reference_summary': [], 'id': [], 'origin': []}
    
    shortlisted_articles, shortlisted_reference_summaries, shortlisted_ids = get_shortlisted_data(articles, reference_summaries, ids )
    shortlisted_data['article'] += shortlisted_articles
    shortlisted_data['reference_summary'] += shortlisted_reference_summaries
    shortlisted_data['id'] += shortlisted_ids
    shortlisted_data['origin'] += ['chemsum'] * len(shortlisted_ids)
            
    isExist = os.path.exists(data_path)
    if not isExist:
        os.makedirs(data_path)
    
    df = pd.DataFrame(shortlisted_data)
    df.to_csv(f'{data_path}/test_sample.csv')
    return df

def make_sample_scitldr(data_path, token_limit = 4096):
    dataset = load_dataset("allenai/scitldr", split = 'test', cache_dir = '/scratch/ramprasad.sa/huggingface_datasets')
    articles = dataset['source']
#     articles = [preprocess_html_tags(each) for each in articles]
    articles = ['\n'.join(each) for each in articles]
    print(articles[0])
    reference_summaries = dataset['target']
    ids = dataset['paper_id']
    
    
    shortlisted_data = {'article': [], 'reference_summary': [], 'id': [], 'origin': []}
    
    shortlisted_articles, shortlisted_reference_summaries, shortlisted_ids = get_shortlisted_data(articles, reference_summaries, ids, dataset = 'scitldr' )
    shortlisted_data['article'] += shortlisted_articles
    shortlisted_data['reference_summary'] += shortlisted_reference_summaries
    shortlisted_data['id'] += shortlisted_ids
    shortlisted_data['origin'] += ['scitldr'] * len(shortlisted_ids)
            
    isExist = os.path.exists(data_path)
    if not isExist:
        os.makedirs(data_path)
    
    df = pd.DataFrame(shortlisted_data)
    df.to_csv(f'{data_path}/test_sample.csv')
    return df

def make_sample_billsum(data_path, token_limit = 4096):
    dataset = load_dataset("billsum", split="test", cache_dir = '/scratch/ramprasad.sa/huggingface_datasets')
    articles = dataset['text']
    reference_summaries = dataset['summary']
    
    
    
    shortlisted_data = {'article': [], 'reference_summary': [], 'id': [], 'origin': []}
    
    shortlisted_articles, shortlisted_reference_summaries, shortlisted_ids = get_shortlisted_data(articles, reference_summaries, dataset = 'billsum')
    shortlisted_data['article'] += shortlisted_articles
    shortlisted_data['reference_summary'] += shortlisted_reference_summaries
    shortlisted_data['id'] += shortlisted_ids
    shortlisted_data['origin'] += ['billsum'] * len(shortlisted_ids)
            
    isExist = os.path.exists(data_path)
    if not isExist:
        os.makedirs(data_path)
    
    df = pd.DataFrame(shortlisted_data)
    df.to_csv(f'{data_path}/test_sample.csv')
    return df

TASK_FUNCMAP = {
    'news':  make_sample_news,
    'pubmed': make_sample_pubmed,
    'chemsum': make_sample_chemsum,
    'scitldr' : make_sample_scitldr,
    'billsum': make_sample_billsum
    
}




if __name__=="__main__":
    
#     python make_samples.py --data-path /home/ramprasad.sa/factual_annotation_llm_summaries/datasets/scitldr --token-limit 4096 --task scitldr
#     python make_samples.py --data-path /home/ramprasad.sa/factual_annotation_llm_summaries/datasets/pubmed --token-limit 4096 --task pubmed 

#     python make_samples.py --data-path /home/ramprasad.sa/factual_annotation_llm_summaries/datasets/pubmed --token-limit 4096 --task pubmed
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--token-limit", type=int, default=4096)
    parser.add_argument("--task", type=str,required = True)
    args = parser.parse_args()
    
    data_path = args.data_path
    token_limit = args.token_limit
    task = args.task

    task_func = TASK_FUNCMAP[task]
    task_func(data_path, token_limit)
    
    
    