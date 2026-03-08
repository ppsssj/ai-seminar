# AI Seminar | Optimization and Validation Strategies

> 2026.02.26 Seminar Review  
> Topics: Momentum, RMSProp, Adam, Validation Data, K-fold Cross Validation

## Overview

이 저장소는 2026년 2월 26일 진행한 AI 세미나 발표 자료와 학습 내용을 정리한 기록입니다.  
이번 세미나에서는 딥러닝 학습 과정에서 중요한 두 축인 **최적화(optimization)** 와 **검증(validation)** 을 중심으로, 다음 다섯 가지 주제를 다뤘습니다.

- Momentum
- RMSProp
- Adam
- Validation Data
- K-fold Cross Validation

발표 자료의 전체 목차도 위 다섯 항목으로 구성되어 있습니다. :contentReference[oaicite:1]{index=1}

이 세미나를 통해 단순히 optimizer의 이름과 공식을 외우는 수준이 아니라,  
**왜 새로운 최적화 기법이 등장했는지**,  
**어떤 학습 병목을 해결하려는지**,  
그리고 **모델 성능을 신뢰할 수 있게 평가하려면 어떤 데이터 분리 전략이 필요한지**를 하나의 흐름으로 정리할 수 있었습니다.

---

## Why This Topic Matters

딥러닝에서 모델 성능은 단순히 네트워크 구조만으로 결정되지 않습니다.  
실제로는 다음 질문에 어떻게 답하느냐가 훨씬 중요합니다.

- loss를 얼마나 안정적으로 줄일 수 있는가?
- 학습이 너무 느리거나, 진동하거나, 발산하지 않는가?
- train 성능이 아니라 실제 unseen data에서도 잘 동작하는가?
- 하이퍼파라미터 선택을 어떤 기준으로 할 것인가?

이번 세미나는 바로 이 문제를 다룹니다.  
앞의 세 주제인 **Momentum, RMSProp, Adam**은 학습을 잘 되게 만드는 방법이고,  
뒤의 두 주제인 **Validation Data, K-fold Cross Validation**은 학습 결과를 믿을 수 있게 만드는 방법입니다.  
즉, **잘 학습시키는 문제와 잘 평가하는 문제를 함께 이해하는 세미나**였습니다. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## 1. Momentum

### 핵심 문제의식

기본 Gradient Descent 계열 방식은 매 스텝마다 **현재 기울기 방향만 보고 이동**합니다.  
문제는 loss surface가 원형이 아니라 길쭉한 타원형 골짜기인 경우입니다. 이때는 가파른 축 방향의 기울기 성분이 크게 나타나서 경로가 좌우로 크게 흔들리고, 최솟값 방향으로는 천천히 내려가게 됩니다. 발표 자료에서도 이런 상황을 “지그재그 발생”과 “손실 최솟점으로 진행이 더딤”으로 설명하고 있습니다. :contentReference[oaicite:5]{index=5}

### 개념 정리

Momentum은 현재 gradient만 사용하는 대신, **이전 이동 방향을 누적해서 현재 업데이트 방향을 결정**하는 방식입니다. 발표 자료에서는 이를 관성으로 설명합니다. 이 접근의 핵심은 다음 두 가지입니다.

- 좌우로 흔들리는 불필요한 성분은 줄인다
- 골짜기 아래 방향의 진행은 더 빠르게 만든다

즉, 방향을 매번 새로 잡지 않고, 이전 이동의 흐름을 이어받아 더 매끈한 경로로 수렴하게 만듭니다. :contentReference[oaicite:6]{index=6}

### 내가 이해한 포인트

Momentum의 본질은 “속도를 붙인다”가 아니라,  
**일관된 방향 성분은 강화하고, 진동성 성분은 상쇄하는 것**입니다.

발표 자료의 설명처럼 현재와 이전 그래디언트를 합산하면, 직전 이동 방향의 영향이 남기 때문에 바로 반대 방향으로 튀는 움직임이 제한됩니다. 그래서 결국 **불필요한 좌우 진동을 줄이고 최저점 수렴을 가속**할 수 있습니다. :contentReference[oaicite:7]{index=7}

### 실무적으로 중요한 이유

Momentum은 특히 다음 상황에서 유효합니다.

- loss surface가 길쭉한 valley 형태일 때
- SGD처럼 업데이트 노이즈가 큰 경우
- 학습 속도를 높이되, 단순 learning rate 증가로 인한 불안정성은 피하고 싶을 때

즉, optimizer를 이해할 때 Momentum은 “첫 번째 개선”으로 봐야 합니다.  
기본 GD의 방향 결정 방식이 너무 단기적이라는 한계를 보완한 것입니다.

---

## 2. RMSProp

### 핵심 문제의식

RMSProp은 “방향”보다 **축별 step size** 문제를 해결하기 위해 등장합니다.

발표 자료에서는 파라미터 벡터 \(w=(a,b)\), 그래디언트 \(g=(0.1,1)\) 예시를 통해 b축의 기울기가 a축보다 10배 큰 상황을 제시합니다. 이 경우 하나의 learning rate \(\eta\)만 사용하면 문제가 발생합니다.

- \(\eta\)를 크게 잡으면 b축이 과도하게 움직여 오버슈팅, 진동, 발산 위험이 커짐
- \(\eta\)를 작게 잡으면 a축은 너무 조금 움직여 학습이 매우 느려짐

즉, **하나의 learning rate로 서로 다른 축을 동시에 최적화하기 어렵다**는 것이 핵심입니다. :contentReference[oaicite:8]{index=8}

### 개념 정리

RMSProp은 각 파라미터의 최근 gradient 크기를 따로 추적합니다.  
발표 자료에서는 각 파라미터의 편미분값을 제곱해서 누적하고, 현재 gradient를 그 누적값의 제곱근으로 나누는 방식으로 설명합니다. 결과적으로:

- gradient가 계속 큰 축은 분모가 커져 step이 자동으로 작아짐
- gradient가 상대적으로 작은 축은 분모가 작아 step이 상대적으로 커짐

즉, **파라미터별로 learning rate를 적응적으로 조절**하는 방식입니다. :contentReference[oaicite:9]{index=9}

### 내가 이해한 포인트

RMSProp의 핵심은 “큰 기울기 축은 조심하고, 작은 기울기 축은 더 과감하게 움직인다”입니다.

발표 자료에서도 가파른 축에서는 오버슈팅과 진동을 줄이고, 완만한 축에서는 평평한 구간을 빠르게 탈출한다고 설명합니다. 이는 단순히 학습을 안정화하는 것뿐 아니라, **파라미터 공간의 비대칭성을 보정하는 장치**라고 볼 수 있습니다. :contentReference[oaicite:10]{index=10}

### 실무적으로 중요한 이유

RMSProp은 다음 상황에서 특히 의미가 있습니다.

- feature scale이나 gradient scale이 축마다 크게 다를 때
- learning rate tuning이 민감한 문제일 때
- 비등방성(non-isotropic) loss surface에서 안정적으로 학습하고 싶을 때

결국 RMSProp은 Momentum과 달리 **방향 개선이 아니라 보폭 조절 문제를 해결한 optimizer**입니다.

---

## 3. Adam

### 핵심 문제의식

Adam은 Momentum과 RMSProp의 장점을 결합한 방식입니다.

발표 자료에서는 Momentum이 방향 안정화와 수렴 가속을 담당하고, RMSProp이 파라미터별 step size 자동 조절을 담당한다고 정리한 뒤, Adam을 **Momentum으로 방향을 만들고 RMSProp 스케일로 나누어 업데이트하는 방식**으로 설명합니다. :contentReference[oaicite:11]{index=11}

### 개념 정리

Adam은 크게 두 가지를 동시에 수행합니다.

- 1차 모멘트 추정: gradient의 방향성과 평균적인 흐름을 반영
- 2차 모멘트 추정: gradient 크기의 분산 정보를 반영하여 step size 조절

발표 자료 표현을 그대로 해석하면, **분자는 Momentum, 분모는 RMSProp**의 역할을 수행합니다. 또한 방향은 관성을 갖게 하고, 보폭은 가파른 축에서는 조심스럽게, 완만한 축에서는 더 과감하게 설정하게 됩니다. :contentReference[oaicite:12]{index=12}

### 내가 이해한 포인트

Adam이 실전에서 널리 쓰이는 이유는 명확합니다.

- Momentum처럼 경로를 안정화한다
- RMSProp처럼 축별로 적절한 step을 선택한다
- 그래서 별도 튜닝 없이도 비교적 빠르게 좋은 초기 성능을 내는 경우가 많다

즉, Adam은 단순히 “좋은 optimizer”라기보다,  
**기존 optimizer들이 각각 해결하던 문제를 하나로 통합한 기본 선택지**에 가깝습니다.

### 실무적으로 중요한 이유

실무에서 Adam이 자주 기본값으로 사용되는 이유는 다음과 같습니다.

- 초반 수렴이 빠른 편
- learning rate 설정이 비교적 덜 민감
- 다양한 파라미터 스케일에 적응적
- 노이즈가 있는 stochastic optimization에서 안정적

다만 개념적으로는 “무조건 Adam이 최고”가 아니라,  
**왜 Adam이 편한지**, 즉 Momentum과 RMSProp의 문제의식을 동시에 품고 있다는 점을 이해하는 것이 더 중요하다고 봤습니다.

---

## 4. Validation Data

### 핵심 문제의식

발표 자료에서는 AI의 진정한 목표를 **훈련 데이터가 아니라 테스트 데이터(실제 데이터)에서 좋은 성능을 보이는 것**이라고 정의합니다. 그런데 여기서 딜레마가 발생합니다.

- 테스트 데이터를 학습 과정에서 사용하면 실제 generalization 성능을 올바르게 파악할 수 없음
- 훈련 데이터만 보면 언제 학습을 멈춰야 할지 판단하기 어려움

즉, **학습 중 의사결정은 필요하지만, test data는 건드리면 안 된다**는 문제가 생깁니다. :contentReference[oaicite:13]{index=13}

### 개념 정리

Validation data는 train data의 일부를 분리해 만든 평가용 데이터입니다.  
중요한 점은 **gradient 계산과 파라미터 업데이트에는 참여하지 않는다**는 점입니다. 발표 자료에서도 “training 데이터의 일부를 validation 데이터로 삼고 기울기 구하는 데에는 참여하지 않는다”고 설명합니다. :contentReference[oaicite:14]{index=14}

또한 train / validation / test의 역할을 다음처럼 비유하고 있습니다.

- 훈련 데이터 → 연습 문제 → 파라미터 학습
- 검증 데이터 → 모의고사 문제 → 하이퍼파라미터 결정
- 테스트 데이터 → 수능 문제 → 최종 성능 측정

이 비유는 세 데이터셋의 역할 차이를 이해하는 데 매우 적절하다고 느꼈습니다. :contentReference[oaicite:15]{index=15}

### 내가 이해한 포인트

Validation data는 단순한 “중간 점검용 데이터”가 아닙니다.  
정확히는 **모델 선택과 학습 종료 시점을 결정하는 기준 데이터**입니다.

발표 자료에서는 train loss는 계속 내려가더라도 val loss는 오히려 다시 올라갈 수 있고, 그 지점부터는 과적합이 시작되었다고 해석할 수 있다고 설명합니다. 또한 hidden layer가 더 많은 모델이 train loss는 잘 줄여도 val loss는 더 빨리 나빠질 수 있음을 예시로 보여줍니다. 반대로 단순한 구조의 모델은 train loss 하강 속도는 느릴 수 있지만 val loss가 더 안정적일 수 있습니다. :contentReference[oaicite:16]{index=16}

즉, validation의 목적은 **훈련 성능 최대화가 아니라 일반화 성능 관점의 선택**입니다.

### 실무적으로 중요한 이유

Validation data가 없으면 다음 판단이 어려워집니다.

- 몇 epoch에서 early stopping 할지
- 어떤 모델 구조가 더 좋은지
- batch size, learning rate, hidden size 같은 하이퍼파라미터를 어떻게 고를지

즉, validation은 “학습 후 평가”가 아니라,  
**학습 중 의사결정 시스템**에 가깝습니다.

---

## 5. K-fold Cross Validation

### 핵심 문제의식

Validation data를 따로 떼어놓는 방식은 유용하지만, 데이터가 적을 때는 문제가 생깁니다.  
발표 자료에서는 전체 데이터 120개 중 훈련 100개, 검증 20개로 나눴을 때, 만약 검증 데이터가 특정 클래스에 편향되어 있으면 하이퍼파라미터 선택도 그 편향을 반영하게 된다고 설명합니다. 예시로 강아지/고양이 분류 문제에서 검증 데이터가 전부 강아지라면 편향된 모델 선택이 이루어질 수 있다고 제시합니다. :contentReference[oaicite:17]{index=17}

### 개념 정리

K-fold cross validation은 데이터를 K개로 나눈 뒤, 각 fold를 번갈아 validation으로 사용하면서 여러 번 학습/검증을 수행하는 방식입니다. 발표 자료에서는 5-fold 예시를 통해:

- 각기 다른 train/validation 조합으로 여러 모델을 학습
- 평균 validation loss를 계산
- 가장 평균 loss가 작은 hyperparameter set을 선택

하는 구조를 설명합니다. 또한 최종적으로는 선택된 하이퍼파라미터로 전체 training data에 대해 다시 학습하거나, 학습된 여러 모델의 출력을 합치는 방식도 사용할 수 있다고 정리합니다. :contentReference[oaicite:18]{index=18}

### 내가 이해한 포인트

K-fold의 핵심은 “데이터가 적을수록, 한 번의 validation split을 너무 믿지 말자”입니다.

즉, 단일 validation split은 우연한 데이터 분포에 크게 흔들릴 수 있으므로,  
여러 split에서 반복적으로 검증해 **평균적인 일반화 성능**을 보는 것이 더 타당합니다.

### 실무적으로 중요한 이유

K-fold는 다음 상황에서 특히 중요합니다.

- 데이터셋 규모가 작을 때
- validation split 편향 가능성이 클 때
- 모델 구조나 하이퍼파라미터 비교의 신뢰도를 높이고 싶을 때

다만 계산 비용은 커지므로,  
항상 쓰는 방식이라기보다 **데이터가 제한된 환경에서 평가 신뢰도를 높이기 위한 전략**으로 이해하는 것이 적절하다고 봤습니다.

---

## What I Learned

이번 세미나를 통해 가장 중요하게 정리한 내용은 다음과 같습니다.

### 1. Optimizer는 “업데이트 공식”이 아니라 “문제 해결 방식”이다

- Momentum은 진동을 줄이고 진행 방향을 안정화한다.
- RMSProp은 축마다 다른 gradient scale 문제를 해결한다.
- Adam은 둘을 결합해 방향과 보폭을 동시에 다룬다.

즉, optimizer는 이름을 외우는 것이 아니라  
**각 방식이 어떤 학습 병목을 해결하려고 나왔는지**로 이해해야 한다고 느꼈습니다. :contentReference[oaicite:19]{index=19} :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

### 2. Train loss만 보고 학습을 판단하면 안 된다

Train loss가 계속 감소해도 validation loss가 다시 증가할 수 있고, 이는 일반화 성능이 나빠지고 있다는 신호일 수 있습니다.  
즉, 학습의 목적은 훈련셋 적합이 아니라 unseen data 일반화입니다. :contentReference[oaicite:22]{index=22} :contentReference[oaicite:23]{index=23}

### 3. 평가 전략도 모델링의 일부다

Validation split을 어떻게 나누느냐에 따라 선택되는 하이퍼파라미터가 달라질 수 있습니다.  
데이터가 적을수록 evaluation protocol 자체가 결과를 좌우하기 때문에, K-fold 같은 검증 전략도 모델 설계의 일부로 봐야 한다고 느꼈습니다. :contentReference[oaicite:24]{index=24}

---

## Practical Takeaways

이번 세미나 내용을 실제 ML/DL 실습에 적용한다면 다음 순서로 활용할 수 있습니다.

1. 기본 optimizer로 Adam을 사용하되, 왜 Adam이 안정적인지 설명할 수 있어야 한다.
2. 학습 로그를 볼 때 train loss만이 아니라 val loss를 함께 보고 early stopping 기준을 잡아야 한다.
3. 데이터가 적은 프로젝트에서는 단일 validation split 결과를 그대로 신뢰하지 말고 cross validation을 고려해야 한다.
4. 모델 구조 선택은 train accuracy가 아니라 validation 기준으로 해야 한다.

---

## Files

- `docs/0226-seminar.pdf` : seminar slides
- `README.md` : seminar summary and study notes

---

## Reflection

이번 세미나는 optimizer와 validation을 각각 분리된 주제로 보는 것이 아니라,  
**모델을 잘 학습시키는 과정과 그 결과를 신뢰성 있게 평가하는 과정이 연결되어 있다**는 점을 이해하는 데 의미가 있었습니다.

특히 다음 흐름이 명확해졌습니다.

- Gradient Descent의 한계가 있다
- 그래서 Momentum이 나왔다
- 축별 learning rate 문제를 해결하기 위해 RMSProp이 나왔다
- 두 장점을 결합한 Adam이 실전 기본 optimizer가 되었다
- 하지만 optimizer를 잘 골라도 평가 전략이 틀리면 일반화 성능을 잘못 해석할 수 있다
- 그래서 validation data와 K-fold cross validation이 중요하다

결국 이번 발표의 핵심은  
**학습 성능을 높이는 기술과, 그 성능을 올바르게 해석하는 방법을 함께 이해하는 것**이었다고 정리할 수 있습니다.
