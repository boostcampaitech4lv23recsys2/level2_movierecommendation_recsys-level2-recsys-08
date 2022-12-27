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

from feature_engineering import FE
from util import inference, make_dataset, make_config

def main(args):
    """모델 train, inference 파일

    args:
        model_name(default - "EASE") : 모델의 이름을 입력받습니다.

        infer(default - True) : 
            True일 경우 submission을 저장합니다.
            inference 과정이 느리기 때문에 필요없다면 False로 하는게 좋습니다.

        dataset_name(default - "train_data) : 데이터셋의 이름을 불러옵니다.

        config_name(default - "basic_config.yaml") : config 정보가 담긴 yaml 파일 이름을 불러옵니다.

        topk(default - 20) : inference를 할 경우에 submission에 유저마다 몇 개의 아이템을 추천할지 정할 수 있습니다.

        나머지는 hyper parameter 입니다. 
    """
    none_neg_list = ["MultiVAE","MultiDAE","MacridVAE","RecVAE","GRU4Rec","NARM","STAMP","NextItNet",\
            "TransRec","SASRec","BERT4Rec","SRGNN","GCSAN","GRU4RecF","FOSSIL","SHAN","RepeatNet","HRM","NPE",]
    
    model_name = args.model_name
    infer = args.inference
    config_name = args.config
    top_k = args.top_k
    dataset_name = args.dataset_name
    del args.__dict__['inference'];del args.__dict__['model_name'];del args.__dict__['config'];del args.__dict__['top_k'];del args.__dict__['dataset_name']

    if not os.path.isdir(f'./dataset/{dataset_name}'):
        print("Make dataset...")
        make_dataset(dataset_name)
    
    if not os.path.isfile(f'./{config_name}'):
        print("Make config...")
        make_config(config_name)

    parameter_dict = args.__dict__
    if model_name in none_neg_list:
        parameter_dict['neg_sampling'] = None
    
    print(f"running {model_name}...")
    result = run_recbole(
        model = model_name,
        dataset = dataset_name,
        config_file_list = [config_name],
        config_dict = parameter_dict,
    )
    print(result)
    wandb.run.finish()

    if infer:
        inference(model_name,top_k)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)