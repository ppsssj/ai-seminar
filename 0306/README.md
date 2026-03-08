# AI Seminar | Unit Step Function, Sigmoid, BCE, Logistic Regression

> 2026.03.06 Seminar Review  
> Topics: Unit Step Function, Sigmoid, Binary Cross Entropy, Logistic Regression

## Overview

이 저장소는 2026년 3월 6일 진행한 AI 세미나 발표 자료와 학습 내용을 정리한 기록입니다.  
이번 세미나에서는 이진 분류(binary classification)를 구성하는 핵심 개념들을 흐름 중심으로 다뤘습니다.

- Unit Step Function
- Sigmoid
- BCE (Binary Cross Entropy)
- Logistic Regression

발표 자료의 목차도 위 네 가지 주제로 구성되어 있습니다. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

이번 세미나의 핵심은 단순히 “시그모이드 함수 공식”이나 “BCE 식”을 외우는 것이 아니라,  
**왜 Unit Step Function만으로는 학습이 어렵고**,  
**왜 Sigmoid가 필요한지**,  
**왜 확률 기반 손실함수(BCE)를 써야 하는지**,  
그리고 **그 구조가 왜 Logistic Regression으로 이어지는지**를 하나의 연결된 맥락으로 이해하는 데 있었습니다.

---

## Why This Topic Matters

이진 분류는 머신러닝과 딥러닝에서 가장 기본적이면서도 중요한 문제입니다.  
예를 들면 다음과 같은 문제들이 모두 이진 분류에 해당합니다.

- 스팸 메일 / 정상 메일
- 강아지 / 고양이
- 합격 / 불합격
- 질병 있음 / 없음

겉으로 보기에는 단순히 0 또는 1을 맞히는 문제처럼 보이지만, 실제 학습 과정에서는 다음 질문이 더 중요합니다.

- 모델이 0과 1을 어떤 기준으로 나누는가?
- 그 기준을 gradient descent로 학습할 수 있는가?
- 출력값을 확률처럼 해석할 수 있는가?
- 예측과 정답의 차이를 어떤 손실함수로 정의할 것인가?

이번 세미나는 바로 이 흐름을 설명합니다.  
즉, **이진 분류의 함수적 기반 → 확률적 해석 → 손실함수 설계 → 모델 해석**까지 이어지는 기초 구조를 다룹니다. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## 1. Unit Step Function

### 핵심 아이디어

발표 자료에서는 키와 몸무게를 입력으로 받아 “빼야 할 사람 / 쪄야 할 사람”처럼 두 클래스로 나누는 예시를 통해 이진 분류를 설명합니다.  
여기서 가장 단순한 형태의 분류 방식은 **입력의 선형 결합 \(Ax + by + c\)** 가 특정 기준보다 큰지 작은지를 보고 0 또는 1을 출력하는 것입니다. :contentReference[oaicite:6]{index=6}

이 구조는 결국 다음 의미를 가집니다.

- 입력을 선형식으로 점수화한다
- 그 점수가 기준을 넘으면 1
- 넘지 못하면 0

즉, 분류 경계는 직선 형태로 나타납니다. 발표 자료에서도 이를  
\(Ax + by + c > 0\),  
\(Y = -(a/b)x - (c/b)\)  
형태로 정리하며, **분류 경계가 선형**임을 설명하고 있습니다. :contentReference[oaicite:7]{index=7}

### 내가 이해한 포인트

Unit Step Function은 이진 분류를 가장 직관적으로 보여주는 함수입니다.  
출력이 0 아니면 1로 딱 끊어지기 때문에 “분류” 자체는 명확합니다.  
하지만 이 명확함이 곧 한계이기도 합니다.

발표 자료에서는 Unit Step Function의 문제점으로 다음 두 가지를 제시합니다.

- 0에서 미분 불가능
- 대부분 구간에서 미분값이 0
- 출력이 0 또는 1로만 나오는 극단적 분류 :contentReference[oaicite:8]{index=8}

즉, 함수는 분류를 잘 정의하지만, **학습에는 적합하지 않습니다.**  
왜냐하면 gradient descent는 미분값을 이용해 파라미터를 업데이트하는데, Unit Step Function은 그 정보가 거의 없기 때문입니다.

### 실무적으로 중요한 이유

이 부분은 “왜 activation function이 단순 임계값 함수면 안 되는가?”를 이해하는 출발점입니다.  
즉, Unit Step Function은 분류 문제의 직관은 제공하지만, **학습 가능한 모델**을 만들기에는 부적합합니다.

---

## 2. Sigmoid

### 왜 필요한가

Unit Step Function의 가장 큰 문제는 미분이 되지 않거나, 미분값이 0이라는 점입니다.  
그래서 발표 자료에서는 이를 “부드럽게 만든 함수”로 Sigmoid를 도입합니다. Sigmoid는 다음 특징을 갖습니다.

- 전 구간에서 미분 가능
- 보다 부드러운 분류 가능
- 출력값을 확률(혹은 정도)로 해석 가능 :contentReference[oaicite:9]{index=9}

즉, Sigmoid는 Unit Step의 “딱 끊기는 결정 경계”를 부드럽게 바꾸면서도, 여전히 0과 1 사이로 출력을 제한합니다.

### 개념 정리

Sigmoid 함수는 다음처럼 정의됩니다.

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

발표 자료에서도 시그모이드를 Unit Step Function의 매끄러운 근사로 설명하고 있습니다.  
입력이 매우 크면 1에 가까워지고, 매우 작으면 0에 가까워지며, 0일 때는 0.5를 출력합니다. :contentReference[oaicite:10]{index=10}

### 내가 이해한 포인트

Sigmoid의 진짜 가치는 “부드럽다”는 데만 있지 않습니다.  
핵심은 **출력값이 확률처럼 해석될 수 있다**는 점입니다.

발표 자료에서는 이미지 분류 예시를 통해, 신경망이 사진을 입력받아 **강아지일 확률**을 출력하도록 만들 수 있다고 설명합니다.  
강아지면 1, 고양이면 0이 되도록 loss를 잘 정의하면 되고, 이때 Sigmoid의 출력값 \(q\)를 확률로 해석합니다. :contentReference[oaicite:11]{index=11}

즉,

- 강아지 사진이면 \(q\)를 크게 만들고 싶다
- 고양이 사진이면 \(q\)를 작게 만들고 싶다
- 또는 \(1-q\)를 크게 만들고 싶다

이렇게 출력 의미를 설계할 수 있습니다. 발표 자료도 이를 “Loss로 출력의 의미를 컨트롤한다”고 표현합니다. :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}

### 실무적으로 중요한 이유

Sigmoid는 단순히 0~1 범위 함수가 아니라,  
**분류 점수를 확률로 변환하는 연결고리**입니다.

이 점을 이해하면 나중에 Logistic Regression, BCE, output layer 설계를 훨씬 자연스럽게 받아들일 수 있습니다.

---

## 3. BCE (Binary Cross Entropy)

### 왜 필요한가

시그모이드를 사용하면 출력값 \(q\)를 확률처럼 해석할 수 있습니다.  
이제 남은 문제는 **이 확률이 정답과 얼마나 다른지 어떻게 수치화할 것인가**입니다.

발표 자료에서는 다음 아이디어로 전개합니다.

- 강아지 사진이면 \(q\)를 최대화
- 고양이 사진이면 \(1-q\)를 최대화 :contentReference[oaicite:14]{index=14}

그리고 미니배치 학습에서는 여러 샘플에 대한 확률을 곱하게 되는데,  
이 값들은 모두 0과 1 사이이므로 계속 곱하면 매우 작아져 **underflow** 문제가 발생할 수 있다고 설명합니다. :contentReference[oaicite:15]{index=15}

그래서 log를 취합니다.  
그 결과:

- 곱이 합으로 바뀌고
- 매우 작은 확률값 계산이 안정화되며
- 최적화가 더 다루기 쉬워집니다 :contentReference[oaicite:16]{index=16}

### 개념 정리

발표 자료에서는 BCE를  
“이진 분류에서 모델의 예측이 정답과 얼마나 다른지 수치화하는 손실함수”라고 정의합니다. :contentReference[oaicite:17]{index=17}

즉, BCE는 다음 목적을 갖습니다.

- 정답이 1일 때는 예측 확률 \(q\)가 커질수록 loss 감소
- 정답이 0일 때는 예측 확률 \(q\)가 작아질수록 loss 감소

결국 모델은 “정답 클래스의 확률을 높이는 방향”으로 학습됩니다.

### 내가 이해한 포인트

BCE는 단순히 공식 암기 대상이 아닙니다.  
핵심은 **확률 모델의 목적함수**라는 점입니다.

즉,

- 모델은 단순 점수가 아니라 확률을 출력하고
- 손실함수는 그 확률이 정답과 얼마나 일치하는지를 측정하며
- log를 통해 수치 안정성과 최적화 편의성을 확보합니다

발표 자료에서는 로그를 취해도 최소화 문제의 방향이 유지된다고 설명하기 위해,  
log가 **단조 증가 함수**라는 점도 같이 설명합니다. 즉, loss를 최소화하는 것과 log(loss)를 최소화하는 것은 같은 방향의 문제입니다. :contentReference[oaicite:18]{index=18}

### 실무적으로 중요한 이유

BCE를 이해하면 다음이 연결됩니다.

- 왜 이진 분류 출력층에서 Sigmoid를 쓰는지
- 왜 MSE보다 BCE가 더 자연스러운지
- 왜 로그 가능도(log-likelihood) 관점으로 손실함수를 볼 수 있는지

즉, BCE는 “이진 분류 전용 손실함수”가 아니라,  
**확률 기반 분류 모델의 핵심 목적함수**로 이해해야 합니다.

---

## 4. Logistic Regression

### 핵심 아이디어

발표 자료에서는 앞에서 본 구조를 “입력과 출력 사이의 관계를 확률 함수로 표현하고, 은닉층이 없는 인공신경망으로 놓고 추정하는 방식”이라고 설명하며 이를 Logistic Regression으로 연결합니다. :contentReference[oaicite:19]{index=19}

즉, Logistic Regression은 다음 구조입니다.

1. 입력을 선형식으로 점수화한다
2. 그 점수를 Sigmoid에 통과시켜 확률로 바꾼다
3. BCE 기반으로 학습한다

발표 자료에서도  
“분류 문제를 선형식으로 점수화한 뒤, 그 점수를 확률로 변환해서 푼다”  
라고 정리하고 있습니다. :contentReference[oaicite:20]{index=20}

### Logit 해석

또한 발표 자료는 Logistic Regression을 **logit을 linear regression으로 구한 것**으로도 해석합니다.  
여기서

- odds = \(q / (1-q)\)
- logit = \(\log(q / (1-q))\)

로 정의하고, 이를 정리하면 다시 Sigmoid 형태가 나온다는 점을 보여줍니다. :contentReference[oaicite:21]{index=21}

이 해석이 중요한 이유는 다음과 같습니다.

- 입력과 logit 사이의 관계는 선형
- 하지만 출력 확률 \(q\)는 비선형
- 즉, Logistic Regression은 **선형 점수 + 비선형 확률 변환** 구조다

발표 자료 마지막 부분에서도  
“사진(입력)과 Logit(출력)의 값이 linear하다”  
“Sigmoid는 BCE를 구하기 위한 확률 변환에 사용된다”  
고 정리합니다. :contentReference[oaicite:22]{index=22}

### 내가 이해한 포인트

“왜 이름이 regression인데 분류 문제에 쓰는가?”가 처음에는 혼동될 수 있습니다.  
하지만 발표 자료 흐름대로 보면 이해가 됩니다.

- 선형 회귀처럼 입력을 선형 결합해 점수를 만든다
- 다만 그 결과를 바로 쓰지 않고 Sigmoid로 확률화한다
- 그 확률을 바탕으로 분류한다

즉, Logistic Regression은  
**회귀적 형태의 점수 함수 위에 확률적 해석을 입힌 분류 모델**입니다.

### 실무적으로 중요한 이유

Logistic Regression은 딥러닝 이전의 고전적 모델이지만, 여전히 중요합니다.

- 이진 분류의 수학적 구조를 가장 명확하게 보여줌
- Sigmoid + BCE 조합의 원리를 이해하게 해줌
- 선형 분류기의 한계와 장점을 동시에 보여줌

즉, 이 모델을 제대로 이해하면 이후의 뉴럴 네트워크 분류기도 더 쉽게 이해됩니다.

---

## What I Learned

이번 세미나를 통해 가장 중요하게 정리한 내용은 다음과 같습니다.

### 1. “분류 가능”과 “학습 가능”은 다르다

Unit Step Function은 분류는 정의할 수 있지만, 미분 불가능하고 gradient 정보가 거의 없어 학습에는 부적합합니다.  
즉, 모델 함수는 출력 형태만이 아니라 **최적화 가능성**까지 고려해야 합니다. :contentReference[oaicite:23]{index=23}

### 2. Sigmoid는 확률 해석을 가능하게 만든다

시그모이드는 단순한 비선형 함수라기보다, 선형 점수를 확률로 연결해주는 함수입니다.  
그래서 이진 분류에서 출력층의 의미를 명확하게 만들어 줍니다. :contentReference[oaicite:24]{index=24} :contentReference[oaicite:25]{index=25}

### 3. BCE는 확률 예측의 오차를 재는 방식이다

정답 클래스의 확률을 높이는 방향으로 학습하게 만드는 손실함수이며, 로그를 통해 곱셈 구조를 합 구조로 바꾸고 수치 안정성을 확보합니다. :contentReference[oaicite:26]{index=26}

### 4. Logistic Regression은 선형 점수 + 확률 변환 구조다

입력과 logit은 선형 관계이지만, 최종 출력은 Sigmoid를 거친 확률입니다.  
즉, 분류 문제를 선형식으로 점수화한 뒤 확률로 해석하는 모델이라고 이해할 수 있습니다. :contentReference[oaicite:27]{index=27}

---

## Practical Takeaways

이번 세미나 내용을 실제 ML/DL 실습에 적용한다면 다음처럼 정리할 수 있습니다.

1. 이진 분류 출력층에서는 “점수”가 아니라 “확률”을 출력한다는 관점으로 설계해야 한다.
2. Unit Step처럼 딱 끊기는 함수보다, gradient 기반 학습이 가능한 매끄러운 함수가 필요하다.
3. 확률 출력에는 BCE 같은 확률 기반 손실함수가 자연스럽게 연결된다.
4. Logistic Regression은 단순 모델이지만, 이진 분류의 구조를 이해하는 가장 좋은 출발점이다.

---

## Files

- `docs/0306-seminar.pdf` : seminar slides
- `README.md` : seminar summary and study notes

---

## Reflection

이번 세미나는 이진 분류를 단편적인 공식 암기가 아니라,  
**함수 선택 → 확률 해석 → 손실함수 설계 → 모델 해석**의 흐름으로 이해하게 해준 발표였습니다.

특히 다음 연결이 가장 중요했습니다.

- Unit Step Function은 직관적이지만 학습이 어렵다
- 그래서 미분 가능한 Sigmoid가 필요하다
- Sigmoid 출력은 확률로 해석할 수 있다
- 그 확률의 차이를 재기 위해 BCE를 사용한다
- 이 구조를 가장 기본적인 분류 모델로 정리한 것이 Logistic Regression이다

결국 이번 발표의 핵심은  
**이진 분류 문제를 수학적으로 어떻게 모델링하고, 그 모델을 학습 가능한 형태로 바꾸는가**를 이해하는 것이었다고 정리할 수 있습니다.
