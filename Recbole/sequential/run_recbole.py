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
import torch

from recbole.quick_start import run_recbole
import wandb
wandb.login()

def run(model_name):
        if model_name in [
            "MultiVAE",
            "MultiDAE",
            "MacridVAE",
            "RecVAE",
            "GRU4Rec",
            "NARM",
            "STAMP",
            "NextItNet",
            "TransRec",
            "SASRec",
            "BERT4Rec",
            "SRGNN",
            "GCSAN",
            "GRU4RecF",
            "FOSSIL",
            "SHAN",
            "RepeatNet",
            "HRM",
            "NPE",
        ]:
            parameter_dict = {
                "train_neg_sample_args" : None,
                "neg_sampling": None,
            }
            return run_recbole(
                model=model_name,
                dataset='train_data',
                config_file_list=['seq.yaml'],
                config_dict=parameter_dict,
            )
        else:
            return run_recbole(
                model=model_name,
                dataset='train_data',
                config_file_list=['seq.yaml'],
            )
            
def main():

    # train load
    train = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")

    # indexing save
    user2idx = {v:k for k,v in enumerate(sorted(set(train.user)))}
    item2idx = {v:k for k,v in enumerate(sorted(set(train.item)))}
    uidx2user = {k:v for k,v in enumerate(sorted(set(train.user)))}
    iidx2item = {k:v for k,v in enumerate(sorted(set(train.item)))}

    train.user = train.user.map(user2idx)
    train.item = train.item.map(item2idx)

    train.columns=['user_id:token','item_id:token','timestamp:float']

    outpath = f"dataset/train_data"
    os.makedirs(outpath, exist_ok=True)
    train.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False)

    yamldata="""
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp

    load_col:
        inter: [user_id, item_id, timestamp]

    show_progress : False
    epochs : 1
    device : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_args:
        split: {'RS': [8, 1, 1]}
        group_by: user
        order: TO
        mode: full
    metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
    topk: 10
    valid_metric: Recall@10
    
    log_wandb : True
    wandb_project : Recbole


    """
    with open("seq.yaml", "w") as f:
        f.write(yamldata)
        
    model_list = ['BERT4Rec'] # 해당 리스트에 쓰고 싶은 모델을 넣어줍니다.
    # os.environ['WANDB_NOTEBOOK_NAME'] = '/opt/ml/input/MR/Recbole/general/Recbole_general.ipynb'
    for model_name in model_list:
        print(f"running {model_name}...")
        start = time.time()
        result = run(model_name)
        t = time.time() - start
        print(f"It took {t/60:.2f} mins")
        print(result)
        wandb.run.finish()

main()