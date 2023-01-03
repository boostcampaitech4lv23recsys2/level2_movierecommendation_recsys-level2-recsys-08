from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"

import os
import json
import argparse
import pandas as pd
import numpy as np
import time, datetime
from tqdm import tqdm
from logging import getLogger
from args import parse_args
import torch

from recbole.quick_start import run_recbole
import wandb
wandb.login()

model_list = ['LR', 'FM', 'NFM', 'DeepFM', 'xDeepFM', 'AFM', 'FNN', 'PNN', 'WideDeep', 'DIN', 'DCN', 'AutoInt']
model_list_needed_feature = ['FFM', 'FwFM'] 


def run(args, model_name):
    """해당 모델에 필요한 config file에 맞춰 run_recbole 리턴"""
    if model_name in model_list:
        return run_recbole(
            model=model_name,
            dataset='train_data',
            config_file_list=['con.yaml'],
            config_dict=args.__dict__,
        )
    elif model_name in model_list_needed_feature:
        return run_recbole(
            model=model_name,
            dataset='train_data',
            config_file_list=['con_sel.yaml'],
            config_dict=args.__dict__,
        )


def main(args):
    """모델 train 파일
    args:
        model_name(default - "FM") : 모델의 이름을 입력받습니다.
        나머지는 hyper parameter 입니다. 
    """

    # train load
    train = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")

    # item 부가 정보 load
    data_path = '/opt/ml/input/data/train'
    year_data = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    writer_data = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
    title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
    genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
    director_data = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')

    # train과 부가 정보 merge
    df_merge = pd.merge(train, year_data, on='item', how='left')
    df_merge = pd.merge(df_merge, writer_data, on='item', how='left')
    df_merge = pd.merge(df_merge, title_data, on='item', how='left')
    df_merge = pd.merge(df_merge, genre_data, on='item', how='left')
    df_merge = pd.merge(df_merge, director_data, on='item', how='left')

    user_data = df_merge[['user']].reset_index(drop=True)
    item_data = df_merge[['item', 'year', 'writer', 'title', 'genre', 'director']].drop_duplicates(subset=['item']).reset_index(drop=True)
    
    # indexing save
    user2idx = {v:k for k,v in enumerate(sorted(set(train.user)))}
    item2idx = {v:k for k,v in enumerate(sorted(set(train.item)))}
    uidx2user = {k:v for k,v in enumerate(sorted(set(train.user)))}
    iidx2item = {k:v for k,v in enumerate(sorted(set(train.item)))}

    # indexing
    train.user = train.user.map(user2idx)
    train.item = train.item.map(item2idx)
    user_data.user = user_data.user.map(user2idx)
    item_data.item = item_data.item.map(item2idx)
    
    # train, item_data 컬럼
    train.columns=['user_id:token','item_id:token','timestamp:float']
    user_data.columns=['user_id:token']
    item_data.columns=['item_id:token', 'year:token', 'writer:token', 'title:token_seq', 'genre:token', 'director:token']
    
    # to_csv
    outpath = f"dataset/train_data"
    os.makedirs(outpath, exist_ok=True)
    train.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False)
    user_data.to_csv(os.path.join(outpath,"train_data.user"),sep='\t',index=False)
    item_data.to_csv(os.path.join(outpath,"train_data.item"),sep='\t',index=False)
    
    # run
    model_name = args.model_name
    print(f"running {model_name}...")
    start = time.time()
    result = run(args, model_name)
    t = time.time() - start
    print(f"It took {t/60:.2f} mins")
    print(result)
    wandb.run.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)