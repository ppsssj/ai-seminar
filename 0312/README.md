# AI Seminar | Batch Normalization, Layer Normalization, Skip-Connection
> 2026.03.12 Seminar Review  
> Topics: Batch Normalization, Layer Normalization, Skip-Connection

## Overview

이 저장소는 2026년 3월 12일 진행한 AI 세미나 발표 자료와 학습 내용을 정리한 기록입니다.  
이번 세미나에서는 딥러닝 학습 과정에서 자주 등장하는 세 가지 핵심 개념인 **Batch Normalization**, **Layer Normalization**, **Skip-Connection**을 중심으로 정리했습니다.

발표 자료의 목차는 다음과 같이 구성되어 있습니다.

- Batch Normalization
- Layer Normalization
- Skip-Connection
- Q&A :contentReference[oaicite:1]{index=1}

이번 세미나의 핵심은 단순히 정규화 기법과 구조를 외우는 것이 아니라,

- 왜 입력값 분포를 조절해야 하는지
- Batch Normalization이 어떤 문제를 해결하려는지
- Batch 기반 방식의 한계를 왜 Layer Normalization이 보완하는지
- 깊은 네트워크에서 왜 Skip-Connection이 필요한지

를 하나의 학습 흐름으로 이해하는 데 있었습니다.

---

## Why This Topic Matters

딥러닝 모델은 층이 깊어질수록 표현력은 높아질 수 있지만, 실제 학습은 오히려 더 어려워질 수 있습니다.  
대표적으로 다음 문제가 발생합니다.

- 활성화 함수 구간에 따라 비선형성이 약해지거나 기울기 흐름이 나빠질 수 있음
- 배치 단위 통계에 의존하는 방식은 작은 batch size에서 불안정할 수 있음
- 네트워크가 깊어질수록 optimization landscape가 복잡해져 학습이 어려워질 수 있음

이번 세미나는 이런 문제를 해결하기 위한 대표 방법으로  
**Batch Normalization → Layer Normalization → Skip-Connection** 흐름을 설명합니다. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## 1. Batch Normalization

### 핵심 문제의식
발표 자료에서는 먼저 특정 노드에 들어가는 값들이 모두 양수이거나 모두 음수일 경우를 예로 들면서,  
입력 분포가 한쪽으로 치우치면 활성화 함수가 충분히 좋은 구간에서 동작하지 못할 수 있다는 점을 설명합니다.  
그래서 입력값들의 **순서는 유지하면서 재배치**하는 아이디어를 제시합니다. :contentReference[oaicite:4]{index=4}

이때 단순히 “평균 0, 분산 1”로 맞추는 정규화만이 항상 최적은 아니라는 점도 함께 짚습니다.  
Sigmoid의 경우 중앙과 바깥 구간의 특성이 다르고, ReLU 역시 무조건 평균 0, 분산 1이 최적이라고 단정할 수 없다고 설명합니다. 즉, **활성화 함수마다 좋은 입력 분포는 다를 수 있다**는 것이 핵심입니다. :contentReference[oaicite:5]{index=5}

### 정규화 과정
발표 자료에서는 예시 값 `1, 2, 3, 4, 5`를 사용해 정규화 과정을 설명합니다.

1. 각 입력값에서 평균을 뺀다
2. 평균이 0이 되도록 이동한다
3. 그 결과를 표준편차로 나눈다
4. 분산과 표준편차가 1이 되도록 만든다 :contentReference[oaicite:6]{index=6}

이 과정은 말 그대로 **입력 분포를 표준화하는 단계**입니다.  
하지만 Batch Normalization의 핵심은 여기서 끝나지 않습니다.

### 학습 가능한 재배치
발표 자료는 정규화된 값을 그대로 쓰는 대신,  
다시 **학습 가능한 두 파라미터**로 조절하는 단계를 설명합니다.

- `a`: 얼마나 스케일할지
- `b`: 어디로 이동시킬지 :contentReference[oaicite:7]{index=7}

즉, 먼저 정규화로 분포를 표준화한 뒤,  
모델이 스스로 “어디에 평균을 둘지”, “얼마나 넓게 퍼뜨릴지”를 다시 학습합니다.  
발표 자료에서는 이를 **비선형성을 살리면서 vanishing gradient 문제와 균형을 맞추는 것**으로 설명합니다. :contentReference[oaicite:8]{index=8}

### Training과 Test의 차이
Batch Normalization은 batch 단위 통계를 사용하기 때문에 training과 test에서 다르게 동작합니다.  
발표 자료에서는 training 시에는 여러 개 입력으로 평균과 분산을 계산할 수 있지만, test 시에는 입력이 1개일 수도 있기 때문에, training 때의 평균과 분산을 moving average 형태로 사용한다고 설명합니다. :contentReference[oaicite:9]{index=9}

또한 batch size가 너무 작으면 현재 batch가 전체 데이터 분포를 잘 대표하지 못해 BN 성능이 흔들릴 수 있다는 점도 단점으로 제시합니다. :contentReference[oaicite:10]{index=10}

### 내가 이해한 포인트
Batch Normalization의 핵심은 단순히 “평균 0, 분산 1 만들기”가 아닙니다.  
정확히는,

- 입력 분포를 일단 안정화하고
- 그 다음 학습 가능한 스케일과 시프트로 다시 적절한 위치에 배치하며
- 활성화 함수의 비선형성과 gradient 흐름 사이에서 더 좋은 균형을 찾는 것

이라고 이해했습니다.

즉, BN은 **정규화 + 재배치 학습**의 결합입니다.

---

## 2. Layer Normalization

### 왜 필요한가
발표 자료에서는 Layer Normalization을 **Batch Normalization의 한계를 보완하는 방식**으로 설명합니다.  
핵심은 BN이 batch size에 영향을 받는다는 점입니다. 반면 LN은 **배치 크기에 영향을 받지 않는다**고 정리합니다. :contentReference[oaicite:11]{index=11}

### 핵심 개념
Layer Normalization은 batch 전체를 기준으로 평균과 분산을 구하는 것이 아니라,  
**한 샘플 내부에서 레이어에 들어가는 값들**을 기준으로 평균과 분산을 계산합니다.  
발표 자료에서도 “레이어에 들어가는 입력값들을 샘플로 사용해 평균과 분산 계산”한다고 설명합니다. :contentReference[oaicite:12]{index=12}

즉, BN과 LN의 차이는 수식 구조보다도  
**무엇을 샘플로 보고 평균·분산을 계산하느냐**에 있습니다.  
발표 자료 역시 “뭘로 평균, 분산 구할 거냐만 다르고 BN과 마찬가지”라고 정리합니다. :contentReference[oaicite:13]{index=13}

### BN과의 차이
Layer Normalization의 주요 특징은 다음과 같습니다.

- Training과 Test에서 동일한 방식으로 계산
- Batch size의 영향을 받지 않음
- 자연어 처리 쪽에서 자주 사용됨 :contentReference[oaicite:14]{index=14}

발표 자료는 특히 `<pad>` 토큰 문제 때문에 NLP에서는 LN이 더 자연스럽게 쓰인다고 설명합니다.  
또한 시각 자료에서는 BN은 같은 채널 내 여러 샘플 방향으로 통계를 보고, LN은 하나의 데이터 안에서 여러 feature 방향으로 통계를 보는 차이를 비교해서 보여줍니다. 13페이지 도식이 이 차이를 직관적으로 설명합니다. :contentReference[oaicite:15]{index=15}

### 내가 이해한 포인트
Layer Normalization은 BN의 대체재라기보다,  
**배치 기반 통계가 불편하거나 불안정한 상황에서 더 적합한 정규화 방식**이라고 보는 것이 맞습니다.

즉,

- 이미지처럼 큰 batch를 쓰기 쉬운 경우 BN이 강력할 수 있고
- NLP처럼 sequence 길이, padding, batch 구성의 영향을 많이 받는 경우 LN이 더 자연스러울 수 있습니다

핵심은 “어느 쪽이 무조건 더 좋다”가 아니라,  
**데이터 구조와 학습 방식에 맞는 정규화 기준을 선택하는 것**입니다.

---

## 3. Skip-Connection

### 핵심 문제의식
발표 자료에서는 BN과 ReLU 조합으로 vanishing gradient 문제를 어느 정도 완화했더라도,  
네트워크가 너무 깊어지면 여전히 학습이 잘 되지 않는 문제가 있다고 설명합니다.  
특히 깊은 모델에서 **loss landscape가 더 꼬불꼬불해지고**, optimization이 어려워진다는 점을 시각 자료와 함께 보여줍니다. :contentReference[oaicite:16]{index=16}

즉, 단순히 정규화만 잘한다고 해서 깊은 네트워크 문제가 전부 해결되는 것은 아닙니다.

### 핵심 개념
Skip-Connection은 매우 간단한 구조입니다.  
발표 자료에서는 **블록을 통과하기 전의 값을 블록 통과 후에 더하는 기법**이라고 설명합니다.  
즉, 블록 출력이 단순히 `f(x)`가 아니라 `x + f(x)`가 됩니다. :contentReference[oaicite:17]{index=17}

여기서 중요한 해석은, 모델이 전체 출력 자체를 처음부터 다 배우는 대신  
**입력과 출력의 차이, 즉 residual만 학습하면 된다**는 점입니다. 발표 자료도 이를 “잔차학습”이라고 설명하며 ResNet을 대표 사례로 제시합니다. :contentReference[oaicite:18]{index=18}

### 왜 효과적인가
Skip-Connection이 효과적인 이유는 다음처럼 이해할 수 있습니다.

- 정보와 gradient가 더 직접적으로 뒤쪽까지 전달된다
- 깊은 네트워크에서도 학습이 덜 막힌다
- 블록이 최소한 identity mapping에 가까운 동작을 하도록 만들 수 있다
- 따라서 깊어진다고 무조건 성능이 망가지는 문제를 줄일 수 있다

발표 자료의 그림에서도 skip connection이 없는 경우 loss landscape가 훨씬 더 거칠고, 있는 경우 훨씬 매끄러운 형태를 보이는 비교가 제시됩니다. :contentReference[oaicite:19]{index=19}

### 내가 이해한 포인트
Skip-Connection의 본질은 단순 지름길이 아닙니다.  
정확히는 **깊은 네트워크가 최소한 원래 입력 정보를 잃지 않도록 보장해주는 구조적 안전장치**에 가깝다고 느꼈습니다.

즉, 블록이 꼭 큰 변환을 학습하지 못하더라도  
최악의 경우 identity에 가까운 동작을 유지할 수 있기 때문에,  
깊은 네트워크 학습의 난이도를 실질적으로 낮춰줍니다.

---

## What I Learned

이번 세미나를 통해 가장 중요하게 정리한 내용은 다음과 같습니다.

### 1. 정규화의 목적은 단순 표준화가 아니다
정규화는 평균 0, 분산 1을 만드는 기계적 연산이 아니라,  
활성화 함수가 더 유리한 구간에서 동작하고 gradient 흐름이 안정되도록 **입력 분포를 조절하는 과정**이라는 점이 핵심이었습니다. :contentReference[oaicite:20]{index=20}

### 2. Batch Normalization은 batch 통계를 쓰기 때문에 batch size 영향이 있다
BN은 강력하지만 training/test 동작 차이와 작은 batch size에서의 불안정성이라는 구조적 한계를 가집니다. :contentReference[oaicite:21]{index=21}

### 3. Layer Normalization은 “무엇을 기준으로 정규화하느냐”를 바꾼 방식이다
LN은 batch 대신 한 샘플 내부 feature를 기준으로 평균과 분산을 계산하기 때문에, batch size에 영향받지 않고 train/test가 동일하게 동작합니다. :contentReference[oaicite:22]{index=22}

### 4. 깊은 네트워크 문제는 정규화만으로 끝나지 않는다
네트워크가 깊어질수록 optimization 자체가 어려워질 수 있고, 이를 완화하는 대표 구조가 Skip-Connection입니다. Residual learning은 깊은 모델 학습을 가능하게 만든 핵심 아이디어라고 느꼈습니다. :contentReference[oaicite:23]{index=23}

---

## Practical Takeaways

이번 세미나 내용을 실제 ML/DL 실습에 연결하면 다음처럼 정리할 수 있습니다.

1. Batch Normalization은 CNN 기반 이미지 모델에서 기본 구성으로 자주 쓰이지만, 작은 batch에서는 성능과 안정성을 같이 확인해야 한다.
2. Layer Normalization은 batch 구성에 덜 의존해야 하는 구조, 특히 NLP 계열에서 더 자연스럽게 활용된다.
3. 깊은 네트워크를 설계할 때는 층을 단순히 쌓는 것보다 residual connection 같은 구조적 장치를 함께 고려해야 한다.
4. 정규화와 skip connection은 각각 다른 문제를 푸는 도구이며, 실제 모델에서는 함께 사용될 수 있다.

---

## Files

- `docs/0312-seminar.pdf` : seminar slides
- `README.md` : seminar summary and study notes

---

## Reflection

이번 세미나는 Batch Normalization, Layer Normalization, Skip-Connection을 각각 따로 외우는 것이 아니라,  
**딥러닝 학습을 더 안정적으로 만들기 위해 어떤 장치들이 등장했는지**를 흐름으로 이해하게 해준 발표였습니다.

특히 다음 연결이 중요했습니다.

- 입력 분포가 활성화 함수에 불리하면 학습이 잘 되지 않을 수 있다
- 그래서 Batch Normalization으로 분포를 조절하고, 그 위치와 스케일까지 학습한다
- 하지만 BN은 batch size에 의존하므로, 이를 보완하는 방식으로 Layer Normalization이 쓰인다
- 그래도 네트워크가 깊어지면 optimization이 어려워지고
- 이를 완화하는 대표 구조가 Skip-Connection과 Residual Learning이다

결국 이번 발표의 핵심은  
**정규화는 분포를 다루는 도구이고, Skip-Connection은 깊이를 다루는 도구**라는 점을 이해하는 것이었다고 정리할 수 있습니다.