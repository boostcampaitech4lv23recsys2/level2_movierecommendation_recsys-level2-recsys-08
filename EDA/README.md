# 탐색적 분석 및 전처리

## A. 데이터 소개
| 파일 | 내용 |
| --- | --- |
| train_ratings.csv | 주 학습 데이터, userid, itemid, timestamp로 구성(5,154,471행) |
| Ml_item2attributes.json | 전처리에 의해 생성된 데이터(item과 genre의 mapping 데이터) |
| title.tsv | 영화 제목(6,807행) |
| years.tsv | 영화 개봉년도(6,799행) |
| directors.tsv | 영화별 감독(한 영화에 여러 감독이 포함될 수 있다, 5,905행) |
| genres.tsv | 영화 장르 (한 영화에 여러 장르가 포함될 수 있다, 15,934행) |
| writers.tsv | 영화 작가 (한 영화에 여러 작가가 포함될 수 있다, 11,307행) |


## B. 데이터 분석 및 Feature Engineering

<img width="633" alt="image" src="https://user-images.githubusercontent.com/57648890/220587039-32154449-5155-4db7-9d26-5a3739558c2a.png">

| Feature | EDA | Feature Engineering |
| --- | --- | --- |
| year | 결측 : 8편의 영화 | title에 있는 연도를 통해 결측치 해결 |
| director, writer | 결측 : 작가 - 1159, 감독 - 1304편의 영화
중복 : 한 영화에 대해 여러명의 작가, 감독이 기여 | 감독, 작가 결측치 → ‘nm0000000’ 카테고리화
감독, 작가 중복정보 → 통계량으로 표현 |
| user | 유저별 그동안 시청한 영화의 평균 interation수가 약 2000에서 10000까지 넓은 범위를 갖는다.
유저별 선호하는 감독 및 작가가 있다. | 유저가 본 아이템의 interaction 수의 평균, 표준편차
유저 선호 정보 피처 추가 ex)유저가 선호하는 감독 3명 |
| genre | 중복 : 한 영화에 대해 최대 10개의 장르정보 | BertTokenizer로 임베딩 후 임베딩 값 평균 |
