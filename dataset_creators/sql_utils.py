import math 
import string 
import re

import json 

import pandas as pd 
import sqlite3



create_table_generated_summaries_str =  '''CREATE TABLE generated_summaries (
    uuid INTEGER PRIMARY KEY AUTOINCREMENT, 
    summary_uuid TEXT NOT NULL ,
    summ_id TEXT NOT NULL, 
    system_id TEXT NOT NULL, 
    summary TEXT NOT NULL,
    article TEXT
);'''

create_table_label_str = '''CREATE TABLE label (
    uuid INTEGER PRIMARY KEY AUTOINCREMENT, 
    user_id TEXT NOT NULL,
    summary_uuid TEXT NOT NULL,
    summ_id TEXT NOT NULL, 
    system_id TEXT NOT NULL,
    label_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    nonfactual_sentences ENUM NOT NULL,
    article TEXT
);'''


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    return conn, c 

def create_table(create_str, db_path):
    conn, c = connect_to_db(db_path)
    c.execute('''%s'''%(create_str))
    conn.commit()
    conn.close()

def add_data(filename, db_path):
    df = pd.read_csv(filename)
    conn, c = connect_to_db(db_path)
#     print('Size before adding' , len(c.execute("""SELECT * from generated_summaries""").fetchall()))
    for idx, row in df.iterrows():
        summ_uuid = row['id']
        article = row['article']
        system_id = row['system_id']
        origin = row['origin']
        
        
        if 'Generic_summary' in df.keys():
            generic_summary = row['Generic_summary']
            summary_uuid_generic = f'{summ_uuid}_{system_id}_gen'
            c.execute("""INSERT INTO generated_summaries (summary_uuid, summ_id, system_id, summary, article) VALUES (?, ?, ?, ?, ?)""",
                                                        (summary_uuid_generic, 
                                                        f'{origin}_generic',
                                                        system_id,
                                                        generic_summary,
                                                        article))
            
            
        if 'Faithful_summary' in df.keys():
            faithful_summary = row['Faithful_summary']
            summary_uuid_faith = f'{summ_uuid}_{system_id}_faith'
            c.execute("""INSERT INTO generated_summaries (summary_uuid, summ_id, system_id, summary, article) VALUES (?, ?, ?, ?, ?)""",
                                                        (summary_uuid_faith, 
                                                        f'{origin}_faithful',
                                                        system_id,
                                                        faithful_summary,
                                                        article))
    
    print('Size after adding' , len(c.execute("""SELECT * from generated_summaries""").fetchall()))
    conn.commit()
    conn.close()
    
def create_tables(db_path, force_new):
    print(force_new)
    conn, c = connect_to_db(db_path)
    create_cmds = {
        'generated_summaries': create_table_generated_summaries_str,
        'label': create_table_label_str
    }
    
    for table_name, create_str in create_cmds.items():
        table_results = c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';").fetchall()
        print(table_results)
        if len(table_results) != 0:
            print('Size before adding' , len(c.execute("""SELECT * from generated_summaries""").fetchall()))
            if (force_new == True):
                print('here', force_new)
                print(f'Dropping {table_name} ...')
                c.execute(f'DROP table {table_name};')
                create_table(create_str, db_path)
        else:
             create_table(create_str, db_path)
            
        
            
    
