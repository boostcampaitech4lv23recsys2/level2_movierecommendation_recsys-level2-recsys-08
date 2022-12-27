import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color

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

def inference(model_name : str,topk : int)->None:
    """
    train.py에서 학습했던 모델로 inference를 하는 함수입니다.
    submission 폴더에 저장됩니다.

    Args:
        model_name (str): 돌렸던 모델의 이름입니다. 해당 모델의 이름이 들어가는 pth파일 중 최근 걸로 불러옵니다.
        topk (int): submission에 몇 개의 아이템을 출력할지 정합니다.
    """
    # model_name이 들어가는 pth 파일 중 최근에 생성된 걸로 불러옴
    save_path = os.listdir('./saved')
    model_path = './saved/' + sorted([file for file in save_path if model_name in file ])[-1]
    K = topk

    # config, model, dataset 불러오기
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    config['dataset'] = 'train_data'

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

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

    # model 평가모드 전환
    model.eval()

    # progress bar 설정
    tbar = tqdm(all_user_list, desc=set_color(f"Inference", 'pink')) # 245, 128

    for data in tbar: # data: 128, 
        # interaction 생성
        interaction = dict()
        interaction = Interaction(interaction)
        interaction[user_id] = data
        interaction = interaction.to(device)

        # user item별 score 예측
        score = model.full_sort_predict(interaction) # [1, 871424]
        score = score.view(-1, item_len) # 128, 6808

        rating_pred = score.cpu().data.numpy().copy() # 128, 6808

        user_index = data.numpy() # 128,

        # idx에는 128명의 영화상호작용이 True, False로 있다.
        idx = matrix[user_index].toarray() > 0 # idx shape: 128, 6808

        rating_pred[idx] = -np.inf # idx에서 True부분이 -inf로 변경
        rating_pred[:, 0] = -np.inf # 첫번째 PAD 열도 -inf로 변경
        
        # np.argpartition(배열, -K) : 배열에서 순서 상관없이 큰 값 K개를 뽑아 오른쪽에 놓겠다 -> 인덱스반환
        # rating_pred에서 각 행마다 K개의 score가 큰 인덱스를 오른쪽에 두고, 그 K개만 가져오기
        ind = np.argpartition(rating_pred, -K)[:, -K:] # rating_pred: (128, 6808) -> ind: (128, 20)

        user_row_index = np.arange(len(rating_pred)).reshape(-1,1) # [[0],[1],...,[127]]
        arr_ind = rating_pred[user_row_index, ind] # 128, 6808 -> 128, 20

        # arr_ind 내부에서 행별로, 내림차순 정렬해서 index 나오도록
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        # ind는 item의 real index를 갖는 128,20 -> arr_ind_argsort를 통해 pred가 높은 상위 20개 read index 추출
        batch_pred_list = ind[user_row_index, arr_ind_argsort] # 128,20 -> 128,20

        if pred_list is None: # 처음에는 직접 정의
            pred_list = batch_pred_list
            user_list = user_index
        else: # pred_list가 있을 때는, append
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(
                user_list, user_index, axis=0
            )

    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))

    # 인덱스 -> 유저 아이템번호 dictionary 불러오기
    with open('./index/uidx2user.pickle','rb') as f:
        uidx2user = pickle.load(f)
    with open('./index/iidx2item.pickle','rb') as f:
        iidx2item = pickle.load(f)
        

    # 데이터 저장
    sub = pd.DataFrame(result, columns=["user", "item"])
    sub.user = sub.user.map(uidx2user)
    sub.item = sub.item.map(iidx2item)

    if not os.path.isdir('./submission'):
        os.mkdir('./submission')

    sub.to_csv(
        f"./submission/{model_path[8:-4]}.csv", index=False # "./saved/" 와 ".pth" 제거
    )

    print('inference done!')
    return