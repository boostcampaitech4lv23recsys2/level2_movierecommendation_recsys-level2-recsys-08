import argparse


def parse_args():
    """
    parameter를 전달해주는 함수입니다.
    sweep을 이용하려면 이 함수에 추가를 하셔야 합니다.
    default 값만 사용하신다면 굳이 추가 안하셔도 됩니다.
    예시로 기본 성능이 좋았던 ~~~ 모델 args를 작성하였습니다.
    일단 대표적인 args 몇가지만 작성했고, 추가로 더 필요한 HP는 추가하셔서 사용하시면 됩니다!
    Returns:
        parser : main에 전달될 args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=10, type=int)

    parser.add_argument("--model_name", default="FM", type=str)

    parser.add_argument("--dataset_name", default="train_data", type=str)

    parser.add_argument("--config",default = "con.yaml",type=str)

    parser.add_argument("--top_k",default = 10,type=int)


    # 공통 
    parser.add_argument("--embedding_size", default=10, type=int) # 전체
    
    parser.add_argument("--mlp_hidden_size", default=(128, 128, 128), type=list) # NFM, DeepFM, xDeepFM, FNN, PNN, WideDeep, DIN, DCN, AutoInt
    
    parser.add_argument("--dropout_prob", default=0.2, type=float) # NFM, DeepFM, xDeepFM, AFM, FNN, PNN, WideDeep, DIN, DCN
    
    
    # xDeepFM
    parser.add_argument("--direct", default=False, type=bool)
    
    parser.add_argument("--cin_layer_size", default=(100, 100, 100), type=list)
    
    
    # AFM, AutoInt
    parser.add_argument("--attention_size", default=25, type=int)
    
    
    # DCN
    parser.add_argument("--cross_layer_num", default=6, type=int)


    # AutoInt
    parser.add_argument("--n_layers ", default=3, type=int)

    parser.add_argument("--num_heads", default=2, type=int)

    parser.add_argument("--dropout_probs ", default=(0.2, 0.2, 0.2), type=list)
    
    
    args = parser.parse_args()

    return args