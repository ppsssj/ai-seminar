# SRCNN 논문 리뷰

> Learning a Deep Convolutional Network for Image Super-Resolution (ECCV 2014)

## 1. Overview
SRCNN은 단일 이미지 초해상도(SISR)를 딥러닝 기반 end-to-end 학습 문제로 정식화한 초기 대표 모델이다. 전통적인 초해상도 파이프라인에서 분리되어 있던 patch extraction, non-linear mapping, reconstruction 과정을 하나의 CNN 안으로 통합했다. 핵심은 **저해상도 이미지를 먼저 bicubic interpolation으로 확대한 뒤**, 그 이미지를 입력으로 받아 더 선명한 고해상도 이미지로 복원하는 매핑을 학습한다는 점이다.

## 2. Why This Paper Matters
SRCNN 이전의 초해상도 방법은 sparse coding, dictionary learning, patch-based matching처럼 여러 단계를 수작업으로 설계하는 방식이 많았다. 이 구조는 전체 파이프라인을 하나의 목적 함수로 직접 최적화하기 어렵다. SRCNN은 이 문제를 CNN 구조로 재해석하면서, 이후 VDSR, SRResNet, EDSR 등으로 이어지는 딥러닝 기반 SISR 연구의 출발점을 만들었다.

## 3. Core Idea
SRCNN의 핵심은 복잡한 전통적 SR 절차를 단순한 3-layer CNN으로 치환한 것이다.

- **입력 전처리**: LR 이미지를 bicubic interpolation으로 먼저 원하는 HR 크기까지 확대
- **특징 추출**: 확대된 이미지에서 선, 경계, 밝기 변화 같은 로컬 특징 추출
- **비선형 매핑**: 추출한 특징을 고해상도 복원에 적합한 표현으로 변환
- **재구성**: 주변 공간 정보를 종합해 최종 HR 이미지 생성

즉, SRCNN은 “작은 이미지를 그냥 키우는 모델”이 아니라, **이미 확대된 이미지에서 부족한 디테일을 보완하는 복원 모델**로 이해하는 것이 정확하다.

## 4. Architecture
SRCNN은 매우 단순한 구조를 가진다.

1. **Conv 9×9, 64 filters**  
   입력 이미지에서 기본적인 특징을 추출한다.
2. **Conv 1×1, 32 filters + ReLU**  
   추출한 특징을 고해상도 복원에 유리한 형태로 변환한다.
3. **Conv 5×5, 1 filter**  
   최종 고해상도 이미지를 재구성한다.

이 구조는 전통적인 SR 절차를 다음처럼 대응시킨다.

- patch extraction → 1st conv
- non-linear mapping → 2nd conv
- reconstruction → 3rd conv

### 모델 흐름
`LR -> Bicubic Upsampling -> Conv(9x9, 64) -> ReLU -> Conv(1x1, 32) -> ReLU -> Conv(5x5, 1) -> SR`

### 레이어 관점에서 보면

| 단계 | 역할 | 직관 |
| --- | --- | --- |
| Bicubic upsampling | 입력을 목표 해상도로 먼저 확대 | 네트워크가 픽셀 위치를 새로 만들지 않고, 이미 커진 이미지에서 디테일 보정에 집중 |
| 1st Conv (9x9) | 패치 특징 추출 | 경계, 선, 밝기 변화 같은 로컬 패턴 추출 |
| 2nd Conv (1x1) | 비선형 매핑 | 추출된 특징을 HR 복원에 적합한 표현으로 변환 |
| 3rd Conv (5x5) | 재구성 | 주변 정보를 종합해 최종 HR 이미지 생성 |

### 왜 1x1 convolution을 쓰는가
SRCNN의 두 번째 층은 새로운 공간 패턴을 넓게 보는 단계라기보다, 첫 번째 층이 뽑아낸 채널별 특징을 다른 표현 공간으로 바꾸는 단계에 가깝다. 그래서 공간 범위를 다시 넓히기보다, 각 위치의 특징 벡터를 재조합하는 `1x1 convolution`이 잘 맞는다.

## 5. Training Objective
SRCNN은 bicubic-upsampled 이미지 `Y`를 입력으로 받아, 원본 고해상도 이미지 `X`에 가까운 출력 `F(Y)`를 학습한다.

핵심 목표는 단순 확대가 아니라,

- 손실된 경계선 복원
- 텍스처 보완
- 원본 HR과의 차이 최소화

즉, **입력과 정답 이미지 사이의 복원 오차를 줄이는 지도학습 기반 회귀 모델**로 볼 수 있다.

보통 학습 목적은 다음처럼 정리할 수 있다.

`Loss = MSE(F(Y), X)`

## 6. Pseudocode
```python
# training
LR, HR = sample_pair()
Y = bicubic_upsample(LR, scale)

f1 = relu(conv9x9(Y, out_channels=64))
f2 = relu(conv1x1(f1, out_channels=32))
SR = conv5x5(f2, out_channels=1)

loss = mse(SR, HR)
update_parameters(loss)
```

```python
# inference
Y = bicubic_upsample(LR, scale)
SR = conv5x5(relu(conv1x1(relu(conv9x9(Y)))))
return SR
```

## 7. Strengths
- 초해상도 문제를 CNN 기반 end-to-end 학습 문제로 정리했다.
- 전통적 sparse-coding 기반 파이프라인을 하나의 네트워크로 통합했다.
- 비교적 단순한 구조로도 bicubic interpolation 및 기존 방법보다 더 높은 복원 성능을 보여주었다.
- 이후 후속 모델들의 구조적 출발점이 되었다.

## 8. Limitations
- 3-layer 구조라 receptive field가 작고, 복잡한 패턴 복원에 한계가 있다.
- 입력 자체가 bicubic interpolation 결과이므로 초기 보간 품질에 의존한다.
- 깊은 네트워크 기반 후속 모델보다 성능이 제한적이다.
- 실제 real-world LR의 blur, noise, sensor distortion을 직접 반영하는 모델은 아니다.

## 9. 발표 때 말하기 좋은 포인트
- SRCNN은 "복잡한 전통적 SR 파이프라인을 CNN 3층으로 통합했다"는 점이 핵심이다.
- 이 모델은 처음부터 HR 전체를 생성하는 느낌보다, bicubic으로 키운 이미지를 더 선명하게 복원하는 모델로 이해하면 쉽다.
- 구조는 단순하지만, 이후 모든 딥러닝 기반 SR의 출발점이라는 상징성이 크다.

## 10. Takeaway
SRCNN의 가장 큰 의의는 성능 숫자 하나가 아니라, **초해상도 문제를 전통적 다단계 파이프라인에서 CNN 기반 end-to-end 복원 문제로 전환했다는 점**이다. 오늘 기준으로 보면 구조는 단순하지만, 연구 흐름 관점에서는 매우 중요한 기준점이다.

## 11. Keywords
`SISR` `SRCNN` `bicubic upsampling` `3-layer CNN` `patch extraction` `non-linear mapping` `reconstruction` `end-to-end learning`

## 12. One-Line Summary
SRCNN은 전통적인 초해상도 파이프라인을 3-layer CNN으로 통합해, SISR을 end-to-end 학습 문제로 바꾼 출발점이다.
