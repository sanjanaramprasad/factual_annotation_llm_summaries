from models.GPT_3.gpt3 import GPTInference
from models.Flan_t5.flan_t5 import FlanT5Inference
import pandas as pd
import dataset_creators.config as config
from tqdm import tqdm
import argparse 
from dataset_creators.sql_utils import connect_to_db
import os

def print_stats(db_path):
    conn, c = connect_to_db(db_path)
    df1 = pd.read_sql("SELECT * from generated_summaries where system_id == 'flant5'", conn)
    df2 = pd.read_sql("SELECT * from generated_summaries where system_id == 'gpt3'", conn)
    df1_articles = list(df1['article'].values)
    df2_articles = list(df2['article'].values)
    assert(df1_articles == df2_articles)
    overlap = len(set(df1_articles).intersection(set(df2_articles)))
    print('Len of database', len(pd.read_sql("SELECT * from generated_summaries", conn)))
    print('Len of overlapping articles in database, for flan, for gpt', overlap, overlap/len(df1), overlap/len(df2))


if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, required=True)
    args = parser.parse_args()
    
    db_path = args.db_path
    print_stats(db_path)
    
    