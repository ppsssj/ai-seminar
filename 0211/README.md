# AI Seminar | Supervised Learning and Self-Supervised Learning
> 2026.02.11 Seminar Review  
> Topics: AI, Supervised Learning, Self-Supervised Learning

## Overview

이 저장소는 2026년 2월 11일 진행한 AI 세미나 발표 자료와 학습 내용을 정리한 기록입니다.  
이번 세미나에서는 AI 학습 방식의 큰 분류를 먼저 살펴본 뒤, 그중 **지도 학습(Supervised Learning)** 과 **자기지도 학습(Self-Supervised Learning)** 의 차이와 활용 방식을 중심으로 정리했습니다.

발표 자료의 목차는 다음과 같이 구성되어 있습니다.

- AI
- 지도 학습
- 비지도 학습
- Q&A :contentReference[oaicite:1]{index=1}

또한 초반 슬라이드에서는 AI/ML/DL의 관계 안에서 지도 학습, 비지도 학습, 자기지도 학습, 강화학습을 함께 비교하는 구조를 제시하고 있습니다. :contentReference[oaicite:2]{index=2}

이번 세미나에서 핵심적으로 다룬 내용은 다음 두 가지였습니다.

- **정답(label)이 있는 데이터를 기반으로 학습하는 지도 학습**
- **정답을 직접 주지 않고 데이터 자체에서 학습 신호를 만들어내는 자기지도 학습**

즉, 이 세미나는 단순히 학습 방식의 정의를 나열하는 것이 아니라,  
**왜 지도 학습이 강력한지**,  
**왜 레이블 비용이 문제가 되는지**,  
**그 한계를 보완하기 위해 자기지도 학습이 어떤 방식으로 등장했는지**를 흐름으로 이해하는 데 의미가 있었습니다.

---

## Why This Topic Matters

머신러닝과 딥러닝을 공부할 때 모델 구조만 보는 경우가 많지만, 실제로는 **학습 데이터를 어떻게 구성하고 어떤 방식으로 학습 신호를 주는가**가 성능과 비용에 큰 영향을 줍니다.

특히 다음 질문이 중요합니다.

- 정답(label)이 있는 데이터는 얼마나 필요한가?
- 라벨링 비용이 클 때는 어떻게 학습해야 하는가?
- 레이블이 부족한 환경에서도 표현을 잘 학습할 수 있는가?
- 실제 downstream task 이전에 어떤 pre-training이 가능한가?

이번 세미나는 바로 이 문제를 다룹니다.  
앞부분의 지도 학습은 **정답 기반 학습**의 구조를 보여주고,  
뒷부분의 자기지도 학습은 **정답을 데이터 내부에서 생성해 사전학습을 수행하는 방법**을 설명합니다. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## 1. Supervised Learning

### 핵심 개념
발표 자료에서는 지도 학습을 **정답(label)을 알고 있는 상태에서 학습하는 방식**으로 설명합니다. 즉, 사람이 입력에 대한 정답을 미리 만들어 놓고, 모델은 그 정답을 맞히는 방향으로 학습합니다. :contentReference[oaicite:5]{index=5}

대표적인 예시로는 다음이 제시됩니다.

- 회귀(regression): 집값, 온도 예측
- 분류(classification): 강아지 vs 고양이 분류 :contentReference[oaicite:6]{index=6}

### 지도 학습이 다루는 대표 문제
이번 발표에서는 지도 학습이 단순 분류만이 아니라 다양한 컴퓨터 비전 문제로 확장된다는 점도 함께 다루고 있습니다.

#### 1) Classification / Localization
한 장의 이미지가 무엇인지 분류하는 작업과, 이미지 안에서 객체의 위치까지 함께 추정하는 작업을 구분해서 설명합니다.  
분류는 이미지 전체를 하나의 클래스로 판단하는 작업이고, 위치 추정은 객체 클래스와 함께 객체의 위치를 출력하는 작업입니다. :contentReference[oaicite:7]{index=7}

#### 2) Object Detection / Segmentation
객체 탐지는 이미지 안의 여러 객체를 동시에 분류하고 위치를 추정하는 작업이고, 분할(segmentation)은 이미지의 모든 픽셀이 어떤 클래스에 속하는지 판단하는 작업으로 정리되어 있습니다. :contentReference[oaicite:8]{index=8}

#### 3) Instance Segmentation
인스턴스 분할은 단순히 픽셀 단위로 클래스를 나누는 것을 넘어, 각 객체 인스턴스를 구분해서 픽셀 단위로 출력하는 문제로 소개됩니다. 발표 자료에서는 “각 픽셀이 어떤 클래스에 속하는지 행렬로 출력”한다고 설명합니다. :contentReference[oaicite:9]{index=9}

#### 4) Pose Estimation / Face Landmarking
자세 추정과 얼굴 랜드마크 탐지도 지도 학습의 대표 응용으로 소개됩니다. 이 슬라이드의 핵심 메시지는 기술 자체보다도, **이런 작업들은 정교한 레이블이 필요하기 때문에 데이터 라벨링에 상당한 시간과 비용이 든다**는 점입니다. :contentReference[oaicite:10]{index=10}

### 내가 이해한 포인트
지도 학습의 장점은 분명합니다.  
정답이 명확하기 때문에 모델이 무엇을 맞혀야 하는지 직접적으로 학습할 수 있고, 성능 평가 기준도 비교적 명확합니다.

하지만 이번 발표에서는 장점보다 **현실적인 한계**가 더 중요하게 드러났습니다.  
즉, 분류, 객체 탐지, 분할, 랜드마킹처럼 문제의 난도가 올라갈수록 필요한 라벨의 양과 정밀도가 커지고, 결국 **데이터 구축 비용이 병목이 된다**는 점입니다. :contentReference[oaicite:11]{index=11}

### 실무적으로 중요한 이유
지도 학습은 지금도 가장 직접적이고 강력한 방식이지만, 실제 프로젝트에서는 다음이 항상 문제입니다.

- 충분한 labeled data 확보
- annotation 품질 관리
- task별 라벨링 비용 증가
- 새로운 도메인으로 갈수록 레이블 수집 난이도 상승

즉, 지도 학습을 이해할 때는 단순히 “정답이 있는 학습”으로 끝내는 것이 아니라,  
**좋은 성능의 대가로 라벨 비용이 크다**는 점까지 함께 봐야 합니다.

---

## 2. Self-Supervised Learning

### 왜 필요한가
발표 자료에서는 지도 학습이 잘 되려면 **정답을 알고 있는 데이터가 많아야 한다**고 설명한 뒤, 곧바로 “But..”을 붙이며 한계를 제시합니다. 핵심은 다음입니다.

- 정답을 만드는 비용이 상당하다
- 문제 상황 자체는 준지도 학습과 유사하다
- 그래서 진짜 풀고 싶은 문제 전에, **가짜 문제(pretext task)** 를 먼저 정의해서 학습한다 :contentReference[oaicite:12]{index=12}

즉, 자기지도 학습은 레이블이 부족한 상황에서 등장한 우회 전략이 아니라,  
**데이터 자체로부터 학습 신호를 만들어 표현을 학습하는 방법**입니다.

### 핵심 개념
발표 자료에서는 자기지도 학습을 “데이터 안에서 self로 만든 정답(label)”을 사용하는 방식으로 설명합니다.  
즉, 사람이 직접 정답을 붙이는 대신, 데이터의 구조나 맥락을 이용해 **스스로 학습용 정답을 생성**합니다. 그래서 이름도 자기지도 학습입니다. :contentReference[oaicite:13]{index=13}

학습 흐름은 다음 두 단계로 제시됩니다.

1. **Pretext task를 학습해서 pre-training**
2. **Downstream task(예: 분류)를 위해 fine-tuning** :contentReference[oaicite:14]{index=14}

이 구조는 매우 중요합니다.  
즉, 자기지도 학습의 목적은 pretext task를 잘 푸는 것 자체가 아니라,  
그 과정을 통해 **좋은 feature representation을 먼저 학습한 뒤**, 실제 task에서 적은 레이블로도 높은 성능을 낼 수 있도록 만드는 것입니다.

---

## 3. Context Prediction

### 개념
발표 자료에서는 자기지도 학습의 첫 번째 예로 **Context Prediction**을 소개합니다.  
핵심 아이디어는 이미지의 픽셀 배치가 완전히 무작위가 아니라, 일정한 구조와 패턴을 가진다는 점을 이용하는 것입니다. :contentReference[oaicite:15]{index=15}

학습 방식은 다음과 같이 설명됩니다.

1. 이미지에서 무작위 위치를 선정해 패치를 생성
2. 기준 패치 주변에 동일한 크기의 패치를 배치
3. 모델이 기준 패치와 주변 패치들의 **상대적 위치 관계**를 예측하도록 학습
4. 이 과정을 통해 모델은 객체 간 위치 관계와 전반적인 구조를 파악하는 능력을 기르게 됨 :contentReference[oaicite:16]{index=16}

### 내가 이해한 포인트
이 방식의 핵심은 단순 위치 맞히기가 아닙니다.  
모델이 상대 위치를 맞히려면 결국 **객체의 형태, 배경, 이미지의 구조적 문맥**을 이해해야 합니다.

즉, Context Prediction은 명시적 라벨 없이도 모델이 이미지의 전반적 구조를 이해하도록 강제하는 pretext task라고 볼 수 있습니다.

### 실무적으로 중요한 이유
이런 방식은 라벨 없이 representation을 학습할 수 있다는 점에서 유용합니다.  
특히 이후 분류, 탐지 같은 downstream task로 넘어갈 때, 완전 랜덤 초기화보다 더 나은 feature extractor를 확보할 수 있다는 점이 중요합니다.

---

## 4. Contrastive Learning

### 개념
발표 자료의 두 번째 자기지도 학습 예시는 **Contrastive Learning**입니다.  
핵심 구조는 하나의 이미지에 서로 다른 두 가지 변형을 적용하고, 이 두 변형이 **같은 원본에서 왔는지 다른 원본에서 왔는지**를 구분하도록 학습하는 방식입니다. :contentReference[oaicite:17]{index=17}

발표 자료는 이 과정을 다음처럼 설명합니다.

1. 하나의 이미지에 서로 다른 두 가지 변형 적용
2. 변형된 이미지 쌍의 출처가 같은지 다른지 인식하며 학습
3. 출처가 같다면 가깝게(Attract), 다르면 멀게 분리
4. 이를 통해 이미지의 고유한 특성과 이미지 간 유사도를 판단하는 능력을 학습 :contentReference[oaicite:18]{index=18}

### 내가 이해한 포인트
Contrastive Learning의 핵심은 분류 정답을 맞히는 것이 아니라,  
**좋은 표현 공간(embedding space)을 만드는 것**입니다.

즉,

- 같은 이미지에서 나온 변형은 가까워져야 하고
- 다른 이미지에서 나온 샘플은 멀어져야 하며
- 그 결과 모델은 이미지의 본질적인 특징을 더 잘 구분하게 됩니다

발표 자료에서도 이 접근이 **적은 레이블만으로도 성능을 끌어올리는 기반**이 된다고 정리하고 있습니다. :contentReference[oaicite:19]{index=19}

### 실무적으로 중요한 이유
Contrastive Learning은 최근 자기지도 학습이 강력해진 대표 이유 중 하나입니다.

- 대규모 unlabeled data 활용 가능
- 적은 labeled data 환경에서 성능 향상 가능
- 사전학습 후 다양한 downstream task로 전이 가능

즉, 이 방식은 “라벨이 없어서 못한다”가 아니라,  
**라벨 없이도 먼저 학습할 수 있다**는 관점을 보여줍니다.

---

## What I Learned

이번 세미나를 통해 가장 중요하게 정리한 내용은 다음과 같습니다.

### 1. 지도 학습의 핵심은 정답 기반 학습이지만, 병목은 라벨링 비용이다
회귀, 분류, 탐지, 분할, 랜드마킹처럼 문제는 다양하지만, 공통적으로 고품질 라벨이 필요합니다.  
특히 task가 복잡해질수록 레이블 비용은 급격히 커집니다. :contentReference[oaicite:20]{index=20}

### 2. 자기지도 학습은 “정답이 없는 학습”이 아니라 “정답을 스스로 만드는 학습”이다
사람이 직접 라벨을 붙이지 않더라도, 데이터의 구조와 관계를 이용해 pretext task를 만들 수 있습니다. 이 점이 비지도 학습과 구별되는 핵심이라고 느꼈습니다. :contentReference[oaicite:21]{index=21}

### 3. 자기지도 학습의 목적은 pretext task 자체가 아니라 representation learning이다
Context Prediction이나 Contrastive Learning은 결국 downstream task 이전에 좋은 feature를 학습하기 위한 수단입니다.  
즉, 사전학습과 파인튜닝 구조를 이해하는 것이 중요합니다. :contentReference[oaicite:22]{index=22}

### 4. 적은 레이블 환경에서 자기지도 학습은 매우 현실적인 대안이 된다
완전한 지도 학습만으로는 데이터 비용이 너무 크기 때문에, unlabeled data를 활용하는 자기지도 학습이 실제로 더 중요해질 수 있다는 점이 인상적이었습니다. :contentReference[oaicite:23]{index=23}

---

## Practical Takeaways

이번 세미나 내용을 실제 ML/DL 학습이나 프로젝트에 연결하면 다음처럼 정리할 수 있습니다.

1. 지도 학습은 성능이 강력하지만, 데이터 라벨링 비용까지 함께 고려해야 한다.
2. 데이터가 충분하지 않거나 라벨 확보가 어려운 문제에서는 자기지도 학습 기반 pre-training을 고려할 수 있다.
3. Context Prediction이나 Contrastive Learning은 단순 예제가 아니라, representation learning의 핵심 아이디어를 보여준다.
4. 실제 모델링에서는 labeled data만이 아니라 unlabeled data를 어떻게 활용할지도 중요한 설계 요소다.

---

## Files

- `docs/0211-seminar.pdf` : seminar slides
- `README.md` : seminar summary and study notes

---

## Reflection

이번 세미나는 지도 학습과 자기지도 학습을 단순 정의 수준으로 구분하는 것이 아니라,  
**왜 자기지도 학습이 필요해졌는지**를 데이터 비용 관점에서 이해하게 해준 발표였습니다.

특히 다음 흐름이 명확했습니다.

- 지도 학습은 정답(label)이 있을 때 강력하다
- 하지만 정답을 만드는 비용이 크다
- 그래서 데이터 자체에서 학습 신호를 만드는 자기지도 학습이 등장했다
- self-generated label 기반 pre-training을 먼저 수행한다
- 이후 downstream task에 fine-tuning하여 적은 레이블로도 성능을 높일 수 있다

결국 이번 발표의 핵심은  
**학습 방식의 차이를 외우는 것이 아니라, 데이터와 레이블의 제약 속에서 어떤 학습 전략이 필요한지를 이해하는 것**이었다고 정리할 수 있습니다.