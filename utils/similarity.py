#!/opt/conda/bin/python
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

def main(args):
    df1 = pd.read_csv(args.a)
    df2 = pd.read_csv(args.b)
    if len(df1)!=len(df2):
        print(f"{args.a}와 {args.b}의 길이가 같아야 합니다.")
    else:
        sim=df1.merge(df2, on=['user','item'])
        sim_ratio = sim.shape[0]/len(df1)
        print(f'겹치는 결과 개수: {sim.shape[0]}')
        print(f'겁치는 결과 비율: {sim.shape[0]} / {len(df1)} = {sim_ratio:.2%}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-a 와 -b 파일의 유사도를 출력합니다.")
    parser.add_argument("-a", default = "/opt/ml/input/code/Recbole/EASE_1_0_Top20_remove_review_after_movie.csv", type=str)
    parser.add_argument("-b", required=True, type=str)
    args = parser.parse_args()
    main(args)