import argparse


def parse_args():
    """
    parameter를 전달해주는 함수입니다.
    sweep을 이용하려면 이 함수에 추가를 하셔야 합니다.
    default 값만 사용하신다면 굳이 추가 안하셔도 됩니다.
    예시로 기본 성능이 좋았던 NextItNet, SASRec, SRGNN, GRU4RecF, SASRecF 모델 args를 작성하였습니다.
    일단 대표적인 args 몇가지만 작성했고, 추가로 더 필요한 HP는 추가하셔서 사용하시면 됩니다!
    Returns:
        parser : main에 전달될 args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=10, type=int)

    parser.add_argument("--model_name", default="SASRec", type=str)

    parser.add_argument("--dataset_name", default="train_data", type=str)

    parser.add_argument("--config",default = "seq.yaml",type=str)

    parser.add_argument("--top_k",default = 10,type=int)


    # 공통 
    parser.add_argument("--embedding_size", default=64, type=int) # NextItNet, SASRec, SRGNN, GRU4RecF
    
    parser.add_argument("--hidden_size", default=64, type=int) # SASRec, GRU4RecF, SASRecF
        
        
    # NextItNet
    parser.add_argument("--kernel_size", default=3, type=int)
    
    parser.add_argument("--block_num", default=5, type=int)
    
    
    # SASRec, SASRecF
    parser.add_argument("--inner_size", default=256, type=int)

    parser.add_argument("--n_layers", default=2, type=int)

    parser.add_argument("--n_heads", default=2, type=int)

    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)

    parser.add_argument("--attn_dropout_prob", default=0.5, type=float)


    # SRGNN
    parser.add_argument("--step", default=1, type=int)

    
    # GRU4RecF
    parser.add_argument("--num_layers", default=1, type=int)

    parser.add_argument("--dropout_prob", default=0.3, type=float)
    
    
    # S3Rec
    parser.add_argument("--pre_model_path", default='', type=str)

    parser.add_argument("--train_stage", default='pretrain', type=str)
    
    
    args = parser.parse_args()

    return args