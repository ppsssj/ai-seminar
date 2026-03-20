# SRGAN 논문 리뷰

> Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (CVPR 2017)

## 1. Overview
SRGAN은 GAN 구조를 단일 이미지 초해상도(SISR)에 적용해, 기존 MSE 기반 복원의 한계였던 **흐릿하고 평균적인 결과**를 개선하려는 모델이다. 핵심은 단순히 픽셀 오차를 줄이는 것이 아니라, **사람이 보기에도 더 자연스럽고 사실적인 고해상도 이미지**를 복원하는 데 있다.

## 2. Why This Paper Matters
기존 SRCNN, VDSR 같은 계열은 PSNR/SSIM 같은 정량 지표에서는 강했지만, 시각적으로는 경계와 질감이 부드럽게 뭉개지는 경우가 많았다. SRGAN은 이 한계를 넘기 위해,

- **Generator**가 HR 이미지를 생성하고
- **Discriminator**가 real/fake를 판별하며
- **Content loss + Adversarial loss**를 함께 사용해

초해상도 평가 기준을 픽셀 정확도에서 **지각 품질(perceptual quality)** 중심으로 확장했다.

## 3. Core Idea
SRGAN의 핵심 메시지는 명확하다.

- 원본과 숫자상 비슷한 이미지보다,
- 사람이 보기에도 진짜 사진처럼 보이는 이미지가 더 중요할 수 있다.

즉, 초해상도 문제를 단순 회귀가 아니라 **복원 + 생성** 문제로 다룬다.

## 4. Architecture
SRGAN은 크게 두 구성요소로 이루어진다.

### Generator
Generator는 저해상도 이미지를 입력받아 고해상도 이미지를 생성한다. 구조적으로는 **SRResNet 기반 residual network**를 사용한다. 즉, SRGAN은 완전히 새로운 복원기가 아니라, residual block 기반 SRResNet 계열을 생성기로 채택하고 여기에 GAN 학습을 결합한 형태다.

### Discriminator
Discriminator는 입력 이미지가 진짜 HR인지, Generator가 만든 fake HR인지 판별하는 분류기다. 세부 구조를 설명하는 모델이 아니라, **real/fake 판별 신호**를 제공하는 역할을 한다. Generator는 이 신호를 활용해 더 자연스럽고 사실적인 질감을 만들도록 학습된다.

### 모델 흐름
`LR -> Generator(SRResNet 기반) -> SR image`

`HR / Generated SR -> Discriminator -> real or fake`

### 구조를 조금 더 풀어 쓰면

| 구성 | 내용 | 역할 |
| --- | --- | --- |
| Generator 입력부 | 초기 convolution | LR 이미지의 기본 feature 추출 |
| Residual blocks | 여러 개의 residual block 반복 | 복원에 필요한 표현 학습 |
| Global skip | 저수준 정보 전달 | 학습 안정화와 정보 보존 |
| Upsampling blocks | sub-pixel convolution 기반 업샘플 | 해상도 확대 |
| 출력부 | 최종 convolution | SR 이미지 생성 |
| Discriminator | CNN 기반 real/fake 분류기 | Generator가 더 자연스러운 질감을 만들도록 압박 |

### SRResNet과의 관계
SRGAN의 Generator는 보통 먼저 **SRResNet 방식으로 pretraining**한 뒤, 그 가중치를 바탕으로 GAN 학습을 진행한다. 그래서 SRGAN은 "완전히 새 모델"이라기보다, **SRResNet + adversarial training + perceptual loss**의 결합으로 이해하면 정리가 쉽다.

## 5. Loss Function
SRGAN의 핵심은 손실 함수 설계다.

### Content Loss
생성 이미지와 원본 HR 이미지가 **내용적으로 얼마나 비슷한지** 반영한다. SRGAN에서는 주로 **VGG feature map 기반 비교**를 사용한다. 즉, 픽셀값 자체보다 구조·경계·질감 같은 고수준 시각 특징 유사성을 더 반영한다.

### Adversarial Loss
생성 이미지가 Discriminator에게 얼마나 **진짜처럼 보이는지** 반영한다. 원본과 픽셀 단위로 직접 비교하는 손실이 아니라, 생성 이미지의 자연스러움을 학습시키는 신호다.

결국 SRGAN은,

- 원본과 내용적으로 비슷해야 하고
- 동시에 진짜처럼 보여야 한다

는 두 조건을 함께 학습한다.

간단히 쓰면 Generator 손실은 다음처럼 볼 수 있다.

`L_G = L_content + lambda * L_adv`

## 6. Training Flow
1. 먼저 Generator를 SRResNet처럼 content loss 중심으로 pretrain한다.
2. 저해상도 이미지를 Generator에 입력해 SR 이미지를 생성한다.
3. 생성 이미지와 원본 HR 이미지 사이의 perceptual/content loss를 계산한다.
4. 생성 이미지와 원본 HR 이미지를 각각 Discriminator에 넣어 real/fake를 판별한다.
5. adversarial loss를 계산한다.
6. Generator는 `content + adversarial` 신호를 함께 받아 업데이트된다.
7. Discriminator는 real/fake 구분 성능을 높이도록 번갈아 학습된다.

## 7. Pseudocode
```python
# step 1: generator pretraining
SR = G(LR)
content_loss = vgg_feature_loss(SR, HR)
update_generator(content_loss)
```

```python
# step 2: adversarial training
SR = G(LR)

loss_content = vgg_feature_loss(SR, HR)
loss_adv_G = -log(D(SR))
loss_G = loss_content + lambda_adv * loss_adv_G

loss_D = -log(D(HR)) - log(1 - D(SR.detach()))

update_discriminator(loss_D)
update_generator(loss_G)
```

## 8. 발표 때 말하기 좋은 구조 요약
- Generator는 "복원기", Discriminator는 "감별자" 역할을 한다.
- SRGAN은 정답 픽셀과의 차이만 줄이는 대신, 사람이 봤을 때 자연스러운 질감까지 함께 학습한다.
- 그래서 PSNR은 조금 손해 볼 수 있어도, 시각적으로는 더 사진 같은 결과가 나올 수 있다.

## 9. Strengths
- 기존 MSE 기반 복원보다 질감과 디테일이 더 자연스럽다.
- 사람이 보기에는 더 사실적인 복원 결과를 제공한다.
- 초해상도 연구에서 perceptual quality의 중요성을 본격적으로 부각했다.
- 이후 ESRGAN 등 perceptual SR 계열 연구의 기반이 되었다.

## 10. Limitations
- PSNR/SSIM 같은 전통적 정량 지표는 오히려 낮아질 수 있다.
- GAN 학습 특성상 학습이 불안정할 수 있다.
- 자연스러워 보이는 질감이 반드시 원본의 실제 디테일과 정확히 일치하는 것은 아니다.
- 복원이라기보다 일부는 생성 성격이 강해질 수 있다.

## 11. Takeaway
SRGAN은 초해상도에서 “정답과 수치적으로 얼마나 가까운가”만 보던 흐름을 바꿔, **실제 사람이 보기에도 자연스럽고 사실적인가**라는 질문을 전면에 올린 모델이다. 즉, SR의 기준을 PSNR 중심에서 perceptual quality 중심으로 확장한 전환점이다.

## 12. Keywords
`SRGAN` `GAN` `Generator` `Discriminator` `SRResNet` `Content Loss` `Adversarial Loss` `VGG Feature Map` `Perceptual Quality`

## 13. One-Line Summary
SRGAN은 SRResNet 기반 Generator와 Discriminator를 함께 학습시켜, 초해상도를 픽셀 정확도 중심 복원에서 지각 품질 중심 복원으로 확장한 모델이다.
