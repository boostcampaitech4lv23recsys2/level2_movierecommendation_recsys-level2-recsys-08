import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color
from recbole.utils.case_study import full_sort_topk
import os

def make_config(config_name : str) -> None:
    yamldata="""
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp

    load_col:
        inter: [user_id, item_id, timestamp]

    show_progress : False
    device : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_args:
        split: {'RS': [9, 1, 0]}
        group_by: user
        order: RO
        mode: full
    metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
    topk: 10
    valid_metric: Recall@10

    log_wandb : True
    wandb_project : Recbole
    """
    with open(f"{config_name}", "w") as f:
        f.write(yamldata)

    return

def make_dataset(dataset_name : str) -> None:
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

    outpath = f"dataset/{dataset_name}"
    os.makedirs(outpath, exist_ok=True)
    # sub_train=train.groupby("user").sample(n=10, random_state=SEED)
    # sub_train.shape
    train.to_csv(os.path.join(outpath,f"{dataset_name}.inter"),sep='\t',index=False)

    return

def uniquify(path:str) -> str:
    """중복파일이 있는 경우 Numbering

    Args:
        path (str): 파일 경로

    Returns:
        str: Numbering된 파일 경로
    """

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + f"-{str(counter)}" + extension
        counter += 1

    return path

def filter_trainset(sub:pd.DataFrame)->pd.DataFrame:
    """train set과 겹치는 Interaction을 필터링합니다.

    Args:
        sub (pd.DataFrame): 필터링 전 submission dataframe

    Returns:
        pd.DataFrame: 필터링 후 submission dataframe
    """
    train = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
    sub = sub.merge(train,on=['user','item'],how='left')
    return sub[sub.time.isna()][['user','item']]

def filter_after_review_interaction(sub:pd.DataFrame) -> pd.DataFrame:
    """마지막 리뷰날짜 이후에 개봉된 영화 Interaction 제외

    Args:
        sub (pd.DataFrame): submission dataframe

    Returns:
        pd.DataFrame: filtering된 submission dataframe
    """
    
    with open('./index/item2year.pickle','rb') as f:
        item2year = pickle.load(f)
    with open('./index/userid2lastyear.pickle','rb') as f:
        userid2lastyear = pickle.load(f)

    sub['lastyear']=sub.user.map(userid2lastyear)
    sub['m_year'] = sub.item.map(item2year)

    sub = sub[sub.lastyear >= sub.m_year]
    return sub[['user','item']]

def inference(model_name : str, topk : int, model_path=None)->None:
    """
    train.py에서 학습했던 모델로 inference를 하는 함수입니다.
    submission 폴더에 저장됩니다.

    Args:
        model_name (str): 돌렸던 모델의 이름입니다. 해당 모델의 이름이 들어가는 pth파일 중 최근 걸로 불러옵니다.
        topk (int): submission에 몇 개의 아이템을 출력할지 정합니다.
    """
    print('inference start!')
    if model_path is None:
        # model_name이 들어가는 pth 파일 중 최근에 생성된 걸로 불러옴
        os.makedirs('saved',exist_ok=True)
        save_path = os.listdir('./saved')
        model_path = './saved/' + sorted([file for file in save_path if model_name in file ])[-1]

    K = topk

    # config, model, dataset 불러오기
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    config['dataset'] = 'train_data'
    config['eval_args']['split']['RS']=[999999,0,1]

    print("create dataset start!")
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print("create dataset done!")

    model = get_model(config['model'])(config, test_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    # device 설정
    device = config.final_config_dict['device']

    # user, item id -> token 변환 array
    user_id = config['USER_ID_FIELD']
    item_id = config['ITEM_ID_FIELD']
    user_id2token = dataset.field2id_token[user_id]
    item_id2token = dataset.field2id_token[item_id]

    # user id list
    all_user_list = torch.arange(1, len(user_id2token)).view(-1,128) # 245, 128

    # user, item 길이
    user_len = len(user_id2token) # 31361 (PAD 포함)
    item_len = len(item_id2token) # 6808 (PAD 포함)

    # user-item sparse matrix
    matrix = dataset.inter_matrix(form='csr') # (31361, 6808)

    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None

    pred_list = None
    user_list = []

    # user id list
    all_user_list = torch.arange(1, len(user_id2token)).view(-1,128) # 245, 128

    tbar = tqdm(all_user_list, desc=set_color(f"Inference", 'pink')) # 245, 128

    for data in tbar:
        batch_pred_list = full_sort_topk(data, model, test_data, K+30, device=device)[1]
        batch_pred_list = batch_pred_list.clone().detach().cpu().numpy()
        if pred_list is None:
            pred_list = batch_pred_list
            user_list = data.numpy()
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(
                user_list, data.numpy(), axis=0
            )
    tbar.close()

    # user별 item 추천 결과 하나로 합쳐주기
    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))

    sub = pd.DataFrame(result, columns=["user", "item"])

    # 인덱스 -> 유저 아이템번호 dictionary 불러오기
    with open('./index/uidx2user.pickle','rb') as f:
        uidx2user = pickle.load(f)
    with open('./index/iidx2item.pickle','rb') as f:
        iidx2item = pickle.load(f)   

    # submission 생성
    sub = pd.DataFrame(result, columns=["user", "item"])
    sub.user = sub.user.map(uidx2user)
    sub.item = sub.item.map(iidx2item)
    sub = filter_trainset(sub)
    sub = filter_after_review_interaction(sub)

    # extract Top K 
    users = sub.groupby('user').user.head(K).reset_index(drop=True)
    items = sub.groupby('user').item.head(K).reset_index(drop=True)
    sub = pd.concat([users,items],axis=1)
    
    print(f"submission length: {sub.shape[0]}")

    os.makedirs('submission',exist_ok=True)
    submission=f"./submission/{model_path[8:-4]}.csv"
    submission = uniquify(submission)
    sub[['user','item']].to_csv(
        submission, index=False # "./saved/" 와 ".pth" 제거
    )
    print(f"model path: {model_path}")
    print(f"submission path: {os.path.relpath(submission)}")
    print('inference done!')
    return