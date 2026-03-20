# VDSR 논문 리뷰

> Accurate Image Super-Resolution Using Very Deep Convolutional Networks (CVPR 2016)

## 1. Overview
VDSR은 초해상도(SISR)에 깊은 CNN을 본격적으로 성공 적용한 대표 모델이다. 핵심은 단순히 네트워크를 깊게 만든 것이 아니라, **20-layer deep CNN**, **residual learning**, **high learning rate**, **adjustable gradient clipping**, **multi-scale 학습**을 결합해 깊은 구조를 실제로 안정적으로 학습시켰다는 점에 있다.

## 2. Why This Paper Matters
SRCNN은 초해상도에 CNN을 적용하는 출발점이 되었지만, 구조가 얕아 문맥 정보 활용이 제한되고 스케일별로 모델을 따로 두는 비효율이 있었다. VDSR은 이 한계를 정면으로 다루며,

- 더 넓은 receptive field 확보
- residual 학습으로 빠른 수렴
- 큰 learning rate를 활용한 빠른 학습
- 하나의 모델로 여러 스케일 처리

라는 방향을 제시했다.

## 3. Core Idea
VDSR의 아이디어는 단순하다.

1. LR 이미지를 먼저 bicubic interpolation으로 원하는 크기까지 키운다.
2. 네트워크는 완성된 HR 전체를 예측하지 않는다.
3. 대신 **입력과 정답의 차이(residual)** 만 학습한다.
4. 최종 결과는 `SR = ILR + R_pred` 로 복원한다.

초해상도에서 입력 ILR은 이미 저주파 정보와 큰 구조를 상당 부분 담고 있다. 따라서 네트워크가 처음부터 전체 이미지를 다시 생성하기보다, **경계선·텍스처·윤곽 같은 고주파 디테일만 보정**하는 편이 더 효율적이다.

## 4. Architecture
VDSR의 구조는 반복적이며 단순하다.

- 입력: bicubic interpolation으로 확대한 ILR 이미지
- 1층: Conv 3×3 + ReLU, 64 channels
- 2~19층: Conv 3×3 + ReLU 반복, 64 channels
- 20층: Conv 3×3, 1 channel residual output
- 모든 층에서 zero-padding 사용

이 설계로 입력과 출력의 spatial size를 유지하면서 깊이를 확보한다.

### 모델 흐름
`LR -> Bicubic Upsampling(ILR) -> [Conv(3x3, 64) + ReLU] x 19 -> Conv(3x3, 1) -> Predicted Residual -> ILR + Residual -> SR`

### 블록 관점에서 보면

| 구성 | 내용 | 역할 |
| --- | --- | --- |
| 입력 | bicubic-upsampled ILR | 저주파 구조를 이미 포함한 기준 이미지 |
| 초반 층 | Conv 3x3 + ReLU | 기본 패턴 추출 |
| 중간 깊은 층 | Conv 3x3 + ReLU 반복 | receptive field 확장, 더 넓은 문맥 반영 |
| 마지막 층 | Conv 3x3 | residual image 예측 |
| skip 형태 복원 | `SR = ILR + R_pred` | 전체 HR 대신 고주파 보정에 집중 |

### 왜 깊은 구조가 필요한가
SRCNN은 구조가 얕아 한 번에 볼 수 있는 문맥 범위가 작았다. VDSR은 작은 `3x3` convolution을 많이 쌓아 receptive field를 넓히고, 더 먼 주변 정보까지 참고해 경계와 텍스처를 복원하려고 한다.

## 5. Key Concepts
### Residual Learning
VDSR은 HR 전체가 아니라 `HR - ILR` 차이만 예측한다.

- 정답 residual: `R_gt = HR - ILR`
- 예측 residual: `R_pred = VDSR(ILR)`
- 최종 출력: `SR = ILR + R_pred`

이 방식은 학습 대상을 단순화하고 수렴을 빠르게 만든다.

### Receptive Field
3×3 convolution을 여러 층 쌓으면 receptive field가 점점 커진다. VDSR은 깊은 구조를 통해 더 넓은 문맥 정보를 반영하고, 그 결과 경계와 구조 복원 성능을 개선한다.

### Adjustable Gradient Clipping
VDSR은 매우 큰 초기 learning rate를 사용한다. 대신 gradient clipping 범위를 learning rate에 맞춰 조절하여 exploding gradient를 방지한다.

- clipping range: `[-θ/γ, θ/γ]`

즉, 학습률을 공격적으로 쓰되 파라미터 업데이트는 안정적으로 유지하는 전략이다.

## 6. Training Flow
1. LR/HR 학습 쌍을 준비한다.
2. LR 이미지를 bicubic interpolation으로 먼저 확대해 ILR을 만든다.
3. 정답 residual `R_gt = HR - ILR`를 계산한다.
4. ILR을 20-layer CNN에 통과시켜 `R_pred`를 얻는다.
5. `SR = ILR + R_pred`로 최종 복원 이미지를 만든다.
6. 손실은 MSE 기반으로 계산한다.  
   `Loss = ||SR - HR||² = ||R_pred - R_gt||²`
7. 큰 learning rate와 gradient clipping을 함께 사용해 학습한다.
8. multi-scale 설정에서는 x2, x3, x4를 한 모델로 함께 학습한다.

## 7. Pseudocode
```python
# training
LR, HR = sample_pair(scale=random_choice([2, 3, 4]))
ILR = bicubic_upsample(LR, scale)
R_gt = HR - ILR

x = relu(conv3x3(ILR, out_channels=64))
for _ in range(18):
    x = relu(conv3x3(x, out_channels=64))

R_pred = conv3x3(x, out_channels=1)
SR = ILR + R_pred

loss = mse(R_pred, R_gt)
update_parameters_with_gradient_clipping(loss)
```

```python
# inference
ILR = bicubic_upsample(LR, scale)
R_pred = vdsr_network(ILR)
SR = ILR + R_pred
return SR
```

## 8. Strengths
- SR에서 깊은 CNN이 실제로 유효하다는 점을 설득력 있게 보여주었다.
- residual learning을 통해 수렴 속도를 크게 높였다.
- 큰 learning rate와 clipping 조합으로 깊은 네트워크 학습을 안정화했다.
- 단일 모델로 multi-scale SR을 처리하는 실용적 방향을 제시했다.
- PSNR/SSIM 기준에서 강한 성능을 보였다.

## 9. Limitations
- bicubic 기반 synthetic SR 가정에 강하다.
- real-world LR의 blur, noise, compression artifact를 직접 반영하는 모델은 아니다.
- MSE/PSNR 중심 복원이라 시각적으로는 다소 부드럽고 보수적인 결과가 나올 수 있다.
- 이후 GAN 계열이 강조한 perceptual realism 측면에서는 한계가 있다.

## 10. 발표 때 말하기 좋은 포인트
- VDSR의 포인트는 "더 깊게 쌓는 것" 자체보다, residual learning으로 깊은 SR 모델을 실제로 학습 가능하게 만든 데 있다.
- bicubic 결과는 큰 구조를 이미 갖고 있으므로, 네트워크는 잔차만 맞추면 된다는 설명이 발표에서 이해가 빠르다.
- SRCNN 다음 단계에서 "깊이"와 "학습 전략"이 왜 중요해졌는지 보여주는 논문으로 연결하기 좋다.

## 11. Takeaway
VDSR의 핵심 공헌은 “깊게 쌓으면 안 된다”가 아니라, **깊은 SR 모델도 올바른 학습 전략이 있으면 충분히 잘 학습된다**는 점을 보여준 것이다. 이후 SRResNet, EDSR, ESRGAN 같은 계열에 이어지는 residual 기반 SR 흐름에서 중요한 전환점이다.

## 12. Keywords
`VDSR` `Very Deep CNN` `Residual Learning` `Receptive Field` `Adjustable Gradient Clipping` `Multi-scale SR` `PSNR` `SSIM`

## 13. One-Line Summary
VDSR은 20-layer deep CNN과 residual learning을 결합해, SRCNN보다 더 깊고 더 빠르게 수렴하는 초해상도 모델을 실현했다.
