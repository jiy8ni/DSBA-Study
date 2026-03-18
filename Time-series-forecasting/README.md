# DSBA Time-series pretraining: Forecasting

## 프로젝트 개요
- 목표: TimesNet 모델을 활용한 Long-term Time-series Forecasting 수행 및 TimesNet 논문에서의 성능 재현

- Dataset: ETT (ETTh1, ETTh2, ETTm1, ETTm2)
- Model: TimesNet
- Framework: PyTorch
- Objective: MSE / MAE 최소화

--- 

## Dataset: ETT
ETT 데이터셋은 전력 변압기 온도 데이터를 기반으로 구성된 장기 시계열 예측 벤치마크 데이터
- ETTh1, ETTh2: 1시간 단위로 측정
- ETTm1, ETTm2: 15분 단위로 측정

| Dataset | Frequency | Length | Description |
|----------|------------|---------|--------------|
| ETTh1 | Hourly | 17,420 | Transformer temperature (1) |
| ETTh2 | Hourly | 17,420 | Transformer temperature (2) |
| ETTm1 | 15-min | 69,680 | Transformer temperature (1) |
| ETTm2 | 15-min | 69,680 | Transformer temperature (2) |

---

## Project Structure
```
.
├── src/
│   ├── models/
│   │   └── TimesNet.py      # TimesNet 및 TimesBlock 아키텍처 구현
│   ├── data_provider/
│   │   └── factory.py       # 데이터 로더 생성 및 관리
│   ├── scripts/
│   │   └── TimesNet/
│   │       └── ETTm1.sh     # ETTm1 데이터셋 실험 실행 스크립트
│   ├── layers/
│   │   ├── Conv_Blocks.py   # Inception Block 등 컨볼루션 레이어
│   │   ├── RevIN            # RevIN 레이어
│   │   └── Embed.py         # Data Embedding 레이어
│   ├── utils/
│   │   ├── metrics.py       # MSE, MAE 등 평가 지표 계산
│   │   └── tools.py         # 하이퍼파라미터 업데이트 도구
│   ├── main.py              # 학습 및 평가 메인 엔트리 포인트
│   └── exp_builder.py       # 실제 학습/검증/테스트 루프 구현
├── requirements.txt         # 필수 라이브러리 목록
└── README.md
```

---

## How to Use

**ETT 데이터셋**
(예시: ETTh1 데이터셋)
```
bash scripts/TimesNet/ETTh1.sh
```

---

## Experimental Settings

[TimesNet 논문](https://arxiv.org/abs/2210.02186)의 Implementation details와 [공식 라이브러리](https://github.com/thuml/Time-Series-Library)를 참고하여 설정하였다.
각 모델에 사용된 학습 설정은 .yaml 파일에서 확인할 수 있다.

### Input / Output 설정

논문에서는 Batch size를 32로 실험하였으나, 본 프로젝트에서는 GPU 여건 상 16으로 설정 후 실험을 진행하였다. 

| Parameter | Value |
|------------|--------|
| Input length (seq_len) | 96 |
| Label length (label_len) | 48 |
| Prediction length (pred_len) | 96 / 192 / 336 / 720 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Epochs | 10 |
| Loss | MSE |

---

## Results (MSE / MAE)

| Dataset  | 96        | 192       | 336       | 720       |
|----------|-----------|-----------|-----------|-----------|
| ETTh1    | 0.5388 / 0.5106 | 0.5758 / 0.5301 | 0.6624 / 0.5519 | 0.7231 / 0.6113 |  
| ETTh2    | 0.2964 / 0.3648 | 0.3328 / 0.3902 | 0.4099 / 0.4414 | 0.4880 / 0.4903 |
| ETTm1    | 0.4771 / 0.4594 | 0.5053 / 0.4713 | 0.5864 / 0.5153 | 0.6348 / 0.5385 |
| ETTm2    | 0.1592 / 0.2572 | 0.2094 / 0.2945 | 0.2603 / 0.3290 | 0.3389 / 0.3816 |  

- pred_len이 길어질 수록 오차 증가
- 특히 pred_len = 720으로 늘렸을 때 오차가 큰 폭으로 증가

| Dataset | Mean | Reported Mean |
|---------|------|---------------|
| ETTh1   | 0.6250 / 0.5509 |0.458 / 0.150 |
| ETTh2   | 0.3818 / 0.4216 |0.414 / 0.427 |
| ETTm1   | 0.5509 / 0.4961 | 0.400 / 0.406 |
| ETTm2   | 0.2420 / 0.3156 | 0.291 / 0.333 |

- ETTm2를 제외하고는 논문 reported 성능에 비해 오차가 큼> learning rate 조절 방식의 차이?


## Results ( RevIN MSE / Original MSE)_0318study

| Dataset  | 96        | 192       | 336       | 720       |
|----------|-----------|-----------|-----------|-----------|
| ETTh1    | 0.5410 / 0.5388 | 0.5757 / 0.5758 | 0.6265 / 0.6624 | 0.7249 / 0.7231 |  
| ETTh2    | 0.2963 / 0.2964 | 0.3335 / 0.3328 | 0.4058 / 0.4099 | 0.4866 / 0.4880 |
| ETTm1    | 0.4754 / 0.4771 | 0.5079 / 0.5053 | 0.5827 / 0.5864 | 0.6324 / 0.6348 |
| ETTm2    | 0.1593 / 0.1592 | 0.2096 / 0.2094 | 0.2572 / 0.2603 | 0.3337 / 0.3389 | 

- TimesNet 기존 모델에서 Instance Normalization을 manually 수행하던 것을 RevIN class를 사용하는 방식으로 코드를 변경한 것이기에, 실험 결과에는 큰 차이가 없었다. 

**RevIN: Reversible Instance Normalization**

RevIN이란, 시계열 데이터에서 발생하는 Distribution Shift(학습 데이터와 테스트 데이터의 통계적 특성이 변화하는 문제)를 해결하기 위한 방법론이다.

시계열 데이터는 시간에 따라 평균과 분산이 변하는 Non-stationary의 성격을 갖는 경우가 많다. 이에 RevIN은 모델 입력 전에 데이터를 정규화하고, 모델의 출력 단계에서 이를 다시 역정규화하여 원래의 스케일로 복원한다. 

- 기존 Instance Normalization과 달리 역변환 과정을 통해 정보 손실을 방지할 수 있음
- 데이터의 경향성(Trend)이나 급격한 변화가 있는 시계열 예측에서 모델의 일반화 성능을 크게 향상시킴
- Trainable Parameters: Scale과 Bias를 학습 가능(learnable)한 파라미터로 설정하여 데이터에 최적화된 정규화를 수행함