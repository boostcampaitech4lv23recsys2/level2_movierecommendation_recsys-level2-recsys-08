import argparse


def parse_args():
    """
    parameter를 전달해주는 함수입니다.

    sweep을 이용하려면 이 함수에 추가를 하셔야 합니다.

    default 값만 사용하신다면 굳이 추가 안하셔도 됩니다.

    예시로 EASE, MultiVAE, MultiDAE, CDAE를 추가하였습니다.

    Returns:
        parser : main에 전달될 args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=1, type=int)

    parser.add_argument("--model_name", default="EASE", type=str)

    parser.add_argument("--dataset_name", default="train_data", type=str)

    parser.add_argument("--inference", default=False, type=lambda s : s.lower() in ['true','1'])

    parser.add_argument("--config",default = "basic_config.yaml",type=str)

    parser.add_argument("--top_k",default = 10,type=int)

    #EASE
    parser.add_argument("--reg_weight", default=250.0, type=float)

    #ADMMSLIM
    parser.add_argument("--lambda1", default=3.0, type=float)

    parser.add_argument("--lambda2", default=200.0, type=float)

    parser.add_argument("--alpha", default=0.5, type=float)

    parser.add_argument("--rho", default=4000.0, type=float)

    parser.add_argument("--k", default=100, type=int)

    # parser.add_argument("--positive_only", default=True, type=bool)

    parser.add_argument("--center_columns", default=False, type=bool)

    #MultiVAE, MultiDAE
    parser.add_argument("--latent_dimendion", default=128, type=int)

    parser.add_argument("--mlp_hidden_size", default=[600],nargs = '+', type=int) # list 형태를 sweep에 적용을 못하겠네요..

    parser.add_argument("--dropout_prob", default=0.5, type=float)
    
    #MultiVAE
    parser.add_argument("--anneal_cap", default=0.2, type=float)

    parser.add_argument("--total_anneal_steps", default=200000, type=int)

    #CDAE
    # parser.add_argument("--loss_type", default="BCE", type=str)

    parser.add_argument("--hid_activation", default="relu", type=str)

    parser.add_argument("--out_activation", default="sigmoid", type=str)

    parser.add_argument("--corruption_ratio", default=0.5, type=float)

    parser.add_argument("--embedding_size", default=64, type=int)

    parser.add_argument("--reg_weight_1", default=0., type=float)

    parser.add_argument("--reg_weight_2", default=0.01, type=float)

    args = parser.parse_args()

    return args