from models.GPT_3.gpt3 import GPTInference
from models.Flan_t5.flan_t5 import FlanT5Inference
import pandas as pd
import dataset_creators.config as config
from tqdm import tqdm
import argparse 
from dataset_creators.sql_utils import create_tables, add_data, connect_to_db
import os
parent_path = '/home/ramprasad.sa/factual_annotation_llm_summaries'
dataset_path = 'datasets/news'

# dataset_path_map = {'xsum': 'datasets/news_sample/xsum_sample.csv', 'cnndm': 'datasets/news_sample/cnndm_sample.csv'}
model_map = {'gpt3' : GPTInference, 'flant5': FlanT5Inference}

def make_sample(data_path, filter_keys = []):
    df = pd.read_csv(f'{data_path}/test_sample.csv')
    if filter_keys:
        print(f'{filter_keys[0]} == {filter_keys[1]}')
        df = df[df[filter_keys[0]] == filter_keys[1]]
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[:num_samples]
    return df

def generate_summaries(df, model, num_samples = 5, instruction_type = 'all'):
    
    model_class = model_map[model]()
    instructions = config.instructions
    
    summaries = {}
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        article = row['article']
        summary = row['reference_summary']
        instruction = instructions[f'{task}_{model}']
        for instr_key, instr in instruction.items():
            if instruction_type == 'all' or instruction_type == instr_key:
                # try:
                summary = model_class.get_news_response(article, instr)
                # except Exception as e:
                # print('Error', e)
                # summary = 'Error'
                if instr_key not in summaries:
                    summaries[instr_key] = []
                summaries[instr_key].append(summary)
    print(summaries.keys())
    for column_key, vals in summaries.items():
        df[column_key] = vals 
    df['system_id'] = [model] * len(df)
    return df




if __name__=="__main__":

#     python create_generated_summaries_database.py --data-path /home/ramprasad.sa/factual_annotation_llm_summaries/datasets/scitldr --num-samples 50 --model flant5 --task scitldr --database-path /home/ramprasad.sa/factual_annotation_llm_summaries/datasets/scitldr/scitldr_summaries_set1.db --force-new-database True
#     python create_generated_summaries_database.py --data-path /home/ramprasad.sa/factual_annotation_llm_summaries/datasets/news --num-samples 25 --model flant5 --task news --database-path /home/ramprasad.sa/factual_annotation_llm_summaries/datasets/news_summaries_set1.db --force-new-database True --instruction-type generic
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--model", type=str,required = True)
    parser.add_argument("--task", type=str,required = True)
    parser.add_argument("--database-path", type=str,default = '')
    parser.add_argument("--force-new-database", type=str, required = True)
    parser.add_argument("--instruction-type", type=str, default = 'all')
    args = parser.parse_args()
    
    data_path = args.data_path
    num_samples = args.num_samples
    model = args.model
    task = args.task
    database_path = args.database_path
    force_new_database = args.force_new_database
    instruction_type = args.instruction_type
    instruction_type = 'Generic_summary' if instruction_type == 'generic' else 'Faithful_summary'
    force_new_database = True if force_new_database == 'True' else False
    
#     df_path = f'{data_path}/{model}_test_sample.csv' 
    df_path_gen_sample = f'{data_path}/{task}_{num_samples}_generation_sample.csv'
    df_path_gen_sample_model_output = f'{data_path}/{model}_{task}_{num_samples}_generation_sample.csv'
    
    if (os.path.exists(df_path_gen_sample) != True) or (force_new_database == True):
        if task == 'news':
                df_xsum = make_sample(data_path, filter_keys = ['origin', 'xsum'])
                df_cnndm = make_sample(data_path, filter_keys = ['origin', 'cnndm'])
                df = pd.concat([df_xsum, df_cnndm])
                df.to_csv(df_path_gen_sample)
        else:
                df = make_sample(data_path)
                df.to_csv(df_path_gen_sample)
                
    else:
        print('sample found')
        df = pd.read_csv(df_path_gen_sample)
                
#     print(os.path.exists(df_path_gen_sample_model_output), df_path_gen_sample_model_output)   
    if (os.path.exists(df_path_gen_sample_model_output) != True or (force_new_database)):
                df = generate_summaries(df, model, num_samples = num_samples, instruction_type= instruction_type)
                print(len(df))
                df = df[df[instruction_type] != 'Error']
                print(len(df))
                df.to_csv(df_path_gen_sample_model_output)
    

        
    if database_path:
        conn, c = connect_to_db(database_path)
        create_tables(database_path, force_new = force_new_database)
        add_data(df_path_gen_sample_model_output, database_path)
        
            
        
    
