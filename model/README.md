# Model

## A. 모델 개요

<img width="927" alt="image" src="https://user-images.githubusercontent.com/57648890/220585185-b11267ff-87aa-4428-99ad-d102acd8ad8b.png">

## B. 모델 선정 밑 분석

이번 대회는 유저가 본 영화들의 implicit feedback(봤다/ 안봤다) 정보를 바탕으로, 유저가 timestamp 기준 순차적으로 본 영화들 중에 Random으로 일부 item을 샘플링한 item 들을 맞추는 대회입니다. 대회 초반에 EDA를 통해 Static한 예측과 Sequenital한 예측 2가지 측면에서 결과를 내어 서로 약점을 보완하는 방향으로 모델을 선정했습니다.

① General

유저별로 중간 중간 Random하게 데이터가 누락된 TrainSet이 주어졌습니다. 데이터 분석 결과 유저의 최소 Interaction이 16개의 영화(item)임을 통해 Cold Start가 어느정도 해결됨을 확인했고, 중간 중간 인터랙션이 비는 것을 채우는 방법에는 AutoEncoder 기반의 General 모델이 좋겠다고 판단되었습니다. 그 결과 EASE, ADMMSLIM, Mult-DAE, CDAE 등 다양한 모델을 학습했고, 성능또한 LB Recall@10 기준 0.14 이상으로 매우 잘 나왔습니다.

② Sequential

하지만, TimeStamp가 주어진 TrainSet을 고려할 때, Sequential 모델역시 General 모델이 고려하지 못하는 아이템을 추천해준다는 판단이 들어, 다양한 Sequentail 모델을 시도했고, 그 결과 GRU4RecF의 성능이 LB Recall@10이 0.1에 가깝게 나왔습니다. General 과 Sequentail모델을 앙상블 하니 성능이 LB Recall@10 기준 0.165가 넘어가며 성능이 크게 올랐습니다.

③ Context-aware

외에도 주어진 정보의 Side Information을 최대한 활용해보자는 판단이 들었습니다. 감독과 작가정보 장르정보가 주어졌기 때문에, 유저별 선호하는 감독, 아이템별 감독 정보를 이용해 Context-Aware 모델을 시도했습니다. 그 결과 FFM, NeuMF에서 LB Recall@10이 0.11에 가깝게 나왔습니다. 유사도 기반으로 이 모델의 결과를 앙상블하니 성능이 LB Recall@10 기준 소수점 3자리 수준에서 상승했습니다.


## C. 모델 분류

<img width="831" alt="image" src="https://user-images.githubusercontent.com/57648890/220584935-7ae4b52b-cc4a-40d1-ab45-dc7bb81f8773.png">

## D. 모델 성능

  | Model | LB Recall@10 |
  | --- | --- |
  | EASE | 0.1595 |
  | ADMMSLIM | 0.1577 |
  | RecVAE | 0.1481 |
  | CDAE | 0.1448 |
  | Mult-DAE | 0.1349 |
  | SLIMElastic | 0.1279 |
  | CatBoost | 0.1261 |
  | FFM | 0.1079 |
  | NeuMF | 0.1048 |
  | ItemKNN | 0.1047 |
  | LGBM | 0.1059 |
  | GRU4RecF | 0.095 |
  
