---
tags: [paper, physiome, original, translation]
source: arXiv:2510.11110v1
---

# PhysioME: 결측 모달리티에 강건한 다중모달 자기지도학습 프레임워크 (한국어 번역)

> 원본: Lee, Cheol-Hui; Lee, Hwa-Yeon; Jung, Min-Kyung; Kim, Dong-Joo. *PhysioME: A Robust Multimodal Self-Supervised Framework for Physiological Signals with Missing Modalities.* arXiv:2510.11110v1 [cs.LG], 2025-10-13.
>
> 저자 소속: 1) 고려대학교 뇌공학과, 2) 고려대학교 정밀공중보건 학제간 협동과정, 3) 고려대학교 의과대학 신경과학교실
> 교신저자: Dong-Joo Kim (dongjookim@korea.ac.kr)

---

## 초록 (Abstract)

생리신호 기반 의료 응용에서는 하드웨어 제약이나 움직임 아티팩트 등으로 인해 일부 모달리티가 결측되거나 손상되는 경우가 흔하다. 그러나 기존 방법 대부분은 모든 모달리티가 항상 사용 가능하다고 가정하기 때문에, 임의의 모달리티가 누락되면 성능이 크게 저하된다. 본 연구는 이 한계를 해결하기 위해, 결측 모달리티 조건에서도 안정적인 성능을 유지하도록 설계된 강건한 프레임워크 **PhysioME**를 제안한다. PhysioME는 다음 세 가지 핵심 요소를 갖는다.

1. 대조학습(contrastive learning)과 마스크 예측(masked prediction)을 결합한 다중모달 자기지도학습(SSL) 방식,
2. 각 생리신호 모달리티의 시간적 동역학을 포착하도록 설계된 **Dual-Path-NeuroNet** 백본,
3. 결측 모달리티 토큰을 복원하여 불완전한 입력을 유연하게 처리할 수 있도록 하는 **복원 디코더(restoration decoder)**.

실험 결과, PhysioME는 다양한 결측 모달리티 시나리오에 걸쳐 높은 일관성과 일반화 성능을 보였다. 이는 PhysioME가 데이터 가용성이 불완전한 실제 임상 환경에서 의사결정을 보조할 수 있는 신뢰성 있는 도구로 활용될 수 있음을 시사한다.

---

## 1. 서론 (Introduction)

실제 임상 환경에서는 센서 고장, 환자의 움직임, 하드웨어 제약 등으로 인해 모든 생리신호를 연속적으로 수집하는 것이 어려운 경우가 많다 (Iranfar et al., 2021). 그러나 기존 방법 대부분은 모든 모달리티가 사용 가능하다는 전제 하에 설계되어, 일부 모달리티가 결측되면 성능이 크게 떨어진다 (Reza et al., 2024).

이러한 한계를 극복하고 환자의 연속 모니터링을 가능하게 하려면, 결측 모달리티 시나리오를 강건하게 처리할 수 있는 모델 구조가 필요하다. 일반적으로 두 가지 전략이 존재한다 (Lee & Kim, 2023; Wu et al., 2024).

- **Dedicated model 접근**: 가능한 모달리티 조합마다 독립된 네트워크를 학습한다. 특정 케이스를 정밀하게 처리할 수 있지만 계산·자원 비용이 매우 크고 실제 환경에 적용하기 어렵다 (Bachmann et al., 2022).
- **Single-model 접근**: 단일 통합 아키텍처가 모든 시나리오를 처리한다. 임상 환경에서의 실용성이 크게 향상되어 최근 활발히 연구되고 있다 (Wu et al., 2024; Bachmann et al., 2022; Lee & Kim, 2023; Deldari et al., 2024; Zhao et al., 2024).

그러나 대부분의 single-model은 비전 태스크에 초점이 맞춰져 있어 (Bachmann et al., 2022; Lee & Kim, 2023; Zhao et al., 2024), 생리신호나 이종 센서로의 확장이 구조적으로 제약된다. 구체적인 한계는 다음과 같다.

1. **시간 동역학 부재**: 비전 모델은 이미지의 공간 정보 처리를 위해 설계되어, 신호의 시간적 연속성을 포착하기 어렵다 (Wang et al., 2020).
2. **결측 의미론 부재**: 결측 신호를 단순한 mask·class 토큰으로 대체하는 방식은 실제 결측 시나리오의 의미를 잘 담지 못해 일반화 성능을 저해한다 (Bachmann et al., 2022; Zhao et al., 2024).
3. **레이블 의존성**: 대부분 지도학습 기반이라 대규모 라벨 데이터를 필요로 하지만, 의료 도메인에서는 라벨이 매우 부족하다 (Esteva et al., 2019; Lee et al., 2024).

이를 해결하기 위해 본 논문은 생리신호에 특화된 single-model 기반 결측 모달리티 프레임워크인 **PhysioME**를 제안한다. 핵심 특성은 다음과 같다.

- **다중모달 SSL 프레임워크**: 대조학습과 마스크 예측을 결합하여 판별적·생성적 표현을 동시에 학습하고, 생리신호에 대한 강건하고 일반화 가능한 특징 표현을 가능하게 한다.
- **DP-NeuroNet 백본**: NeuroNet (Lee et al., 2024) 구조를 공유하는 두 개의 병렬 경로에 서로 다른 시간적 증강(temporal augmentation)을 적용하여, 풍부한 모달리티별 표현을 학습한다.
- **복원 디코더 모듈**: 모달리티마다 별도의 복원 디코더를 두고, 관측된 모달리티 정보를 활용해 결측 모달리티 토큰을 재구성한다. 이를 통해 다양한 결측 시나리오를 강건하게 처리할 수 있다.

PhysioME는 임상 데이터셋에서 수면 단계 분류와 저혈압 예측 두 가지 다운스트림 태스크에 대해, 다양한 결측 모달리티 조건에서도 안정적인 성능을 보인다.

---

## 2. 방법론 (Methods: PhysioME)

### 2.1 DP-NeuroNet

DP-NeuroNet은 동일한 NeuroNet 인스턴스를 가중치 공유 형태로 두 개 두는 구조이다 (Figure 2). NeuroNet (Lee et al., 2024)은 마스크 예측과 대조학습을 결합해 생리신호로부터 일반화 가능한 표현을 비라벨 데이터로 학습하는 SSL 프레임워크이다.

NeuroNet은 다중 스케일 1D ResNet 기반의 frame network와 ViT 기반의 encoder–decoder로 구성된다. frame network가 만든 토큰 시퀀스의 일부를 무작위로 샘플링한 뒤 encoder–decoder를 통해 복원하며, 복원 손실은 MSE로 최적화된다. 또한 두 개의 독립적으로 샘플링된 토큰 시퀀스에 대해 NT-Xent 손실 (Sohn, 2016) 기반의 대조학습을 적용한다 (NeuroNet의 자세한 구조는 Appendix I 참고).

증강된 두 입력 $\tilde{x}_i^{(1)}, \tilde{x}_i^{(2)}$ 는 각각 NeuroNet으로 처리되어 표현 $z^{(1)}, z^{(2)}$ 가 추출된다. 표현은 NeuroNet 인코더 $f_{\text{neuronet enc}}$ (frame network + ViT 인코더)와 MLP 레이어를 통해 얻어지며, 동일 입력의 두 증강 뷰 간 유사도가 커지도록 NT-Xent 손실이 부여된다.

$$
\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2B}\sum_{b=1}^{B}\bigl[\ell(z_b^{(1)}, z_b^{(2)}) + \ell(z_b^{(2)}, z_b^{(1)})\bigr]
$$

$$
\ell(z_a, z_b) = -\log\frac{\exp(\text{sim}(z_a, z_b)/\tau)}{\sum_{k=1}^{2B}\mathbb{1}_{[k\neq a]}\exp(\text{sim}(z_a, z_k)/\tau)}
$$

여기서 $B$는 배치 크기, $\text{sim}(\cdot)$은 코사인 유사도이다.

### 2.2 PhysioME 본체

PhysioME는 결측 모달리티 처리를 위해 특별히 설계된 다중모달 SSL 프레임워크이다 (Figure 1). 핵심 구성요소는 다음과 같다.

#### (A) 학습 과정 (Training)

**모달리티 인코더 (Modality Encoder).**  각 생리신호는 사전학습된 DP-NeuroNet 인코더 $f_{\text{neuronet enc}}$ 를 모달리티 인코더로 사용해 부호화된다. 총 $M$개의 모달리티 인코더가 있고, $m$번째 인코더는 $m$번째 모달리티 특징을 추출한다 ($m \in \{1, \ldots, M\}$). 입력 $x^{(m)}$ 에 대해 인코더는 길이 $N$의 토큰 시퀀스 $e^{(m)} = \{e_i^{(m)}\}_{i=1}^N = f_{\text{modality enc}}^{(m)}(x^{(m)}) = f_{\text{neuronet enc}}(x^{(m)})$ 를 출력한다. PhysioME 적응을 위해 일부 레이어에는 LoRA (Hu et al., 2022)를 적용하고 나머지는 동결한다.

**다중모달 인코더 (Multimodal Encoder).**  $M$개의 토큰 시퀀스 $e^{(m)}$ 은 ViT 기반 다중모달 인코더에 입력되어 통합 토큰 시퀀스를 생성하며, 다음 세 단계로 구성된다.

1. **Positional·Modality Encoding**: 각 $e^{(m)}$ 을 MLP에 통과시킨 뒤 위치 인코딩 $pe$ 와 모달리티 인코딩 $mt^{(m)}$ 을 더한다.
   $z^{(m)} = \text{MLP}(e^{(m)}) + pe + mt^{(m)}$
2. **Drop Modality / Random Sampling**: 각 $z^{(m)}$ 에 대해 두 가지 중 하나가 무작위로 적용된다.
   - **Drop modality**: 토큰 시퀀스 $z^{(m)}$ 전체를 제거하고 마스크 토큰 $\mu$ 로 치환하여 결측 시나리오를 시뮬레이션.
   - **Random sampling**: $z^{(m)}$ 에서 $\tilde{N} (< N)$ 개 토큰을 인덱스 집합 $\mathcal{H}$ 에 따라 추출하여 $\tilde{z}^{(m)}$ 구성.
3. **Fusion Token**: 결과 시퀀스 $\tilde{h}$ 를 다중모달 인코더 $f_{\text{multimodal enc}}$ 에 입력하여, 샘플링된 fusion 토큰 시퀀스 $\tilde{o}$ 를 생성한다.

**모달리티 디코더 (Modality Decoder).**  ViT 기반 모달리티 디코더는 모달리티별로 독립 구성된다. 각 디코더는 drop modality 가 적용되지 않은 모달리티 집합 $\tilde{\mathcal{M}}$ 에 해당하는 부분 토큰 $\tilde{o}^{(m)}$ 만을 입력으로 받는다. 이때 random sampling 에서 선택되지 않은 인덱스 위치에는 마스크 토큰 $\mu$ 를 삽입한 후, 디코더 $f_{\text{modality dec}}^{(m)}$ 와 MLP 를 거쳐 새 시퀀스 $d^{(m)} = \text{MLP}(f_{\text{modality dec}}^{(m)}(\mu, \tilde{o}^{(m)} \mid m \in \tilde{\mathcal{M}}))$ 를 만든다. $d^{(m)}$ 은 원본 모달리티 인코더 출력 $e^{(m)}$ 을 복원하도록 MSE 로 최적화된다. SSL 학습이 끝나면 모달리티 디코더는 폐기된다.

**복원 디코더 (Restoration Decoder).**  ViT 기반 구조이며 모달리티 디코더보다 깊게 설계되어 정밀 복원을 가능하게 한다. 학습 시 stop-gradient 가 적용되어 다중모달 인코더 등 상위 네트워크로 역전파가 차단되며, 복원 디코더는 오직 복원 태스크에만 집중하여 독립적으로 학습된다.

복원 디코더 $f_{\text{restoration dec}}^{(m)}$ 은 drop modality 가 적용된 모달리티들 ($m \in \mathcal{M} \setminus \tilde{\mathcal{M}}$) 의 토큰 $\tilde{o}^{(m)}$ 만 입력으로 받아, 새 시퀀스 $g^{(m)} = \text{MLP}(f_{\text{restoration dec}}^{(m)}(\tilde{o}^{(m)} \mid m \in \mathcal{M}\setminus\tilde{\mathcal{M}}))$ 을 생성한다. 이는 원본 인코더 출력 $e^{(m)}$ 을 MSE 로 복원하도록 학습된다. 학습 종료 후 복원 디코더는 추론 단계에서 결측 모달리티의 토큰 시퀀스를 추정하는 데 사용된다.

#### (B) 목적 손실 (Objective Loss)

1. **재구성 손실 (Reconstruction Loss)**: 모달리티 디코더 출력 $d^{(m)}$ 은 drop 되지 않은 모달리티 $m \in \tilde{\mathcal{M}}$ 에서만 정의된다. random sampling 에서 선택되지 않은 토큰 인덱스 $\bar{\mathcal{H}}^{(m)}$ 위치에서만 MSE 손실을 계산한다.
   $$
   \mathcal{L}_{\text{intra recon}} = \frac{1}{|\tilde{\mathcal{M}}|}\sum_{m \in \tilde{\mathcal{M}}}\frac{1}{|\bar{\mathcal{H}}^{(m)}|}\sum_{i \in \bar{\mathcal{H}}^{(m)}}\bigl\|d_i^{(m)} - e_i^{(m)}\bigr\|_2^2
   $$

2. **결측 모달리티 재구성 손실 (Missing Modality Reconstruction Loss)**: 복원 디코더는 drop 된 모달리티 $m \in \mathcal{M}\setminus\tilde{\mathcal{M}}$ 에 대해서만 학습된다.
   $$
   \mathcal{L}_{\text{missing recon}} = \frac{1}{|\mathcal{M}\setminus\tilde{\mathcal{M}}|}\sum_{m \in \mathcal{M}\setminus\tilde{\mathcal{M}}}\frac{1}{N}\sum_{i=1}^N\bigl\|g_i^{(m)} - e_i^{(m)}\bigr\|_2^2
   $$

3. **교차 모달리티 대조 손실 (Cross-Modality Contrastive Loss)**: 모달리티 인코더 표현 $e^{(m)}$ 과 다중모달 인코더의 fusion 표현 $o$ 사이의 정합성을 위해 NT-Xent 기반 대조학습을 적용한다. 두 표현은 시퀀스 차원으로 평균 → 정규화 → MLP 통과 과정을 거쳐 각각 $\bar{e}^{(m)}$, $\bar{o}$ 가 된다.
   $$
   \mathcal{L}_{\text{NT-Xent}}^{(m)} = \frac{1}{2B}\sum_{b=1}^{B}\bigl[\ell(\bar{e}_b^{(m)}, \bar{o}_b) + \ell(\bar{o}_b, \bar{e}_b^{(m)})\bigr]
   $$
   $$
   \ell(z_a, z_b) = -\log\frac{\exp(\text{sim}(z_a, z_b)/\tau)}{\sum_{k=1}^{2B}\mathbb{1}_{[k\neq a]}\exp(\text{sim}(z_a, z_k)/\tau)}
   $$
   최종 교차 모달리티 손실은 모든 모달리티에 대한 평균이다.
   $$
   \mathcal{L}_{\text{cross contra}} = \frac{1}{|\mathcal{M}|}\sum_{m \in \mathcal{M}}\mathcal{L}_{\text{NT-Xent}}^{(m)}
   $$

4. **최종 손실 (Final Loss)**: 위 세 손실을 가중합하여 학습한다.
   $$
   \mathcal{L} = \alpha\,\mathcal{L}_{\text{intra recon}} + \beta\,\mathcal{L}_{\text{missing recon}} + \gamma\,\mathcal{L}_{\text{cross contra}}
   $$
   여기서 $\alpha, \beta, \gamma$ 는 세 손실의 비중을 조절하는 하이퍼파라미터이다.

#### (C) 추론 (Inference)

일부 모달리티가 결측된 시나리오를 처리하기 위해 PhysioME는 다음 절차를 따른다.

1. **관측 모달리티 인코딩**: 관측된 모달리티 $\mathcal{M}_{\text{obs}} \subseteq \{1,\ldots,M\}$ 에 대해 입력 $x^{(m)}$ 을 모달리티 인코더와 MLP 에 통과시킨 뒤 $pe$, $mt^{(m)}$ 를 더한다.
   $$
   e^{(m)} = f_{\text{modality enc}}^{(m)}(x^{(m)}),\quad m \in \mathcal{M}_{\text{obs}}
   $$
   $$
   z^{(m)} = \text{MLP}(e^{(m)}) + pe + mt^{(m)}
   $$
2. **다중모달 인코딩**: 토큰 $z^{(m)}$ 을 마스크 토큰 $\mu$ 와 결합해 다중모달 인코더에 투입한다.
   $$
   o = f_{\text{multimodal enc}}(\text{concat}(\mu, z^{(m)} \mid m \in \mathcal{M}_{\text{obs}}))
   $$
3. **결측 모달리티 토큰 복원**: 결측 모달리티 $\mathcal{M}_{\text{miss}} = \{1,\ldots,M\}\setminus\mathcal{M}_{\text{obs}}$ 에 해당하는 $o^{(m)}$ 을 추출하여 복원 디코더에 통과시킨다.
   $$
   g^{(m)} = \text{MLP}(f_{\text{restoration dec}}^{(m)}(o^{(m)})),\quad m \in \mathcal{M}_{\text{miss}}
   $$
4. **Fusion Token 생성**: 복원된 토큰과 관측 토큰을 합쳐 다중모달 인코더에 다시 입력하여 최종 토큰 시퀀스 $ft$ 를 얻는다. 여기에는 전체 시퀀스를 대표하는 class token 이 포함되며 다운스트림 태스크의 최종 예측에 사용된다.
   $$
   ft = f_{\text{multimodal enc}}\bigl(\text{concat}(g^{(m)} \mid m \in \mathcal{M}_{\text{miss}},\ e^{(m)} \mid m \in \mathcal{M}_{\text{obs}})\bigr)
   $$

---

## 3. 실험 (Experiments)

### 3.1 데이터셋

- **Sleep-EDFX** (Kemp et al., 2000). 폴리솜노그래피(PSG) 기록을 포함하며, 미국수면의학회(AASM) 기준에 따라 다섯 개 수면 단계로 분류한다. 본 연구에서는 SC 서브셋과 EEG Fpz-Cz, EEG Pz-Oz, 수평 EOG 의 3채널을 사용했고, 모든 신호에 0–40 Hz 대역통과 필터를 적용했다.
- **VitalDB** (Lee et al., 2022). 6,388건의 수술 케이스에서 수집한 수술 중 생리신호 기록과 주산기 환자 정보를 포함하며, 486,451개의 파형 세그먼트로 구성된다. 본 연구에서는 동맥혈압(ABP), 심전도(ECG), 광혈류측정(PPG) 신호를 활용해 향후 5분 이내의 수술 중 저혈압을 예측한다.

### 3.2 베이스라인

결측 모달리티를 다루는 다중모달 SSL 프레임워크 4종을 베이스라인으로 구현했다. CroSSL 을 제외한 모델들은 원래 비전 태스크용이므로, 신호를 단시간 푸리에 변환(STFT) 이미지로 변환해 입력으로 사용했다.

- **MultiMAE** (Bachmann et al., 2022): MAE 기반의 다중모달 마스크 입력 복원.
- **RobustSsF** (Lee & Kim, 2023): 모달리티별로 4개의 독립 encoder–decoder 분기와 coupled regularization 을 사용한 SSL.
- **CroSSL** (Deldari et al., 2024): 잠재 공간(latent) 마스킹을 통해 이종 센서 모달리티의 글로벌 표현을 학습.
- **MaskMentor** (Zhao et al., 2024): 완전 입력으로 학습한 teacher 와 부분 결측 입력의 student 가 교사 예측을 의사 라벨로 받는 마스크 자기교육 프레임워크.

### 3.3 평가 방식

5-fold subject-group cross-validation 을 사용했다. 데이터셋은 pretrain, train, test 세 부분으로 분할했고, pretrain 서브셋은 라벨 없이 SSL 에 사용, train 서브셋은 제한된 라벨로 linear evaluation 에 사용했다. 사전학습된 네트워크에 다운스트림 분류기를 부착해 두 다운스트림 태스크를 수행하고, test 서브셋에서 성능을 측정했다.

### 3.4 평가 지표

정확도 (ACC) 와 ROC 곡선 아래 면적 (AUC) 으로 평가했다. AUC 는 수면 단계 분류와 저혈압 예측처럼 클래스 불균형이 있는 태스크에 적합한 지표다.

$$
\text{ACC} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR})\,d\text{FPR},\quad \text{TPR} = \frac{TP}{TP+FN},\ \text{FPR} = \frac{FP}{FP+TN}
$$

---

## 4. 결과 (Results)

### 4.1 다른 방법론과의 비교

두 가지 다운스트림 태스크에 대해 평가했다: (1) Sleep-EDFX 를 이용한 수면 단계 분류 (Table 1), (2) VitalDB 를 이용한 저혈압 예측 (Table 2). 비교 대상은 single-model 계열 (MultiMAE, RobustSsF, CroSSL, MaskMentor) 과 dedicated-model 계열 (ContraWR, SynthSleepNet) 이며, 다양한 모달리티 조합 실험으로 결측 모달리티 시나리오에서의 강건성도 평가했다.

### 4.2 Single-model 내부 비교

**(결측 모달리티 하 성능)** PhysioME 는 광범위한 결측 모달리티 시나리오에서 기존 방법들을 일관되게 상회했다. 두 다운스트림 태스크 모두에서 모든 모달리티 결측 조합에 걸친 ACC·AUC 평균이 가장 높았다. 수면 단계 분류 태스크에서는 단일 모달리티 EEG Pz-Cz 케이스를 제외한 거의 모든 시나리오에서 1위였으며 (Table 1 굵은 표기), 저혈압 예측 태스크에서는 단일 모달리티 ECG·PPG 에서 일부 모델이 ACC 가 더 높았으나, AUC 는 모든 시나리오에서 최고치를 기록했다 (Table 2 굵은 표기).

**(결측 모달리티 하 강건성)** PhysioME 는 대부분의 결측 시나리오에서 full-modality 대비 성능 저하가 가장 작았다 (Table 1·2 파란색 강조). full-modality 대비 평균 절대 차이(MAV) 기준으로, 수면 단계 분류 ACC 3.97% / AUC 2.04, 저혈압 예측 ACC 3.54% / AUC 5.43 로 모든 지표에서 가장 작은 저하를 보였다. 즉, PhysioME 는 결측 조건에서도 안정적인 성능을 유지한다.

### 4.3 Single-model 과 Dedicated-model 비교

**(태스크 특화 모델 대비 경쟁력)** PhysioME 는 single-model 임에도 불구하고 두 다운스트림 태스크의 다양한 모달리티 조합에서 dedicated-model 과 동등하거나 더 나은 성능을 보였다 (Table 1·2 밑줄 강조). 이는 dedicated-model 이 본질적으로 더 우월하다는 통념에 반하는 결과이며, single-model 만으로도 강건하고 신뢰할 수 있는 성능 달성이 가능함을 보여준다.

구체적으로, 수면 단계 분류에서는 모든 모달리티 조합에서 ContraWR 을 능가하고 SynthSleepNet 도 대부분의 케이스에서 상회했다. 저혈압 예측에서는 dedicated-model 들과 동급의 성능을 냈으며, 특히 AUC 측면에서 단일 ECG 케이스를 제외한 모든 시나리오에서 ContraWR 을 앞섰다. SynthSleepNet 이 전체 AUC 는 더 높았지만, ABP+PPG 와 ECG+PPG 조합에서는 PhysioME 가 우세했다.

### 4.4 절제 연구 (Ablation Studies)

Sleep-EDFX 데이터셋에서 PhysioME 의 최적 구성을 찾기 위한 절제 연구를 수행했다. 학습 효율을 위해 10 epoch 으로 제한하고, 세 가지 모달리티 설정에서 평가했다: (i) EEG Fpz-Cz, (ii) EEG Fpz-Cz + EOG, (iii) EEG Fpz-Cz + EEG Pz-Cz + EOG.

**모달리티 백본 네트워크.**  Table 3 결과, DP-NeuroNet 백본이 전반적으로 가장 높은 성능을 보였다. CNN 기반 (ResNet, EfficientNet) 이 ViT 기반 (ViT, Swin Transformer) 보다 우수했고, DP-NeuroNet 은 모든 모달리티 조합에서 원본 NeuroNet 을 상회했다. 단일 모달리티 설정에서는 ACC 6.09%, AUC 3.22 만큼 NeuroNet 을 앞섰다.

| 모델 | Modality 1 ACC/AUC | Modality 2 ACC/AUC | Full ACC/AUC |
|---|---|---|---|
| ResNet | 70.13 / 89.12 | 71.69 / 89.80 | 74.66 / 90.97 |
| EfficientNet | 68.66 / 86.21 | 72.01 / 88.54 | 74.38 / 89.53 |
| ViT | 42.68 / 66.08 | 47.16 / 66.99 | 48.18 / 65.28 |
| Swin Transformer | 51.22 / 73.76 | 55.95 / 75.79 | 57.19 / 75.65 |
| NeuroNet | 70.42 / 89.48 | 74.43 / 91.23 | 75.90 / 92.02 |
| **DP-NeuroNet** | **76.51 / 92.70** | **77.77 / 93.85** | **78.83 / 94.46** |

(Table 3 재구성)

**복원 디코더 그래디언트의 유무.**  복원 디코더의 그래디언트를 상위 네트워크로 역전파할지 여부를 검토했다. Table 4 결과, stop-gradient 를 적용해 그래디언트 전파를 차단하는 편이 모든 모달리티 조합에서 일관되게 우수했다. 이는 복원 디코더를 격리해 결측 모달리티 토큰 복원에만 집중하도록 하는 전략이 전반적인 성능을 끌어올림을 시사한다.

**복원 전략.**  세 가지 대안을 비교했다 (Table 5): (i) masked tokens, (ii) memory tokens (선행 연구에서는 modality·learnable token 등으로 표기됨; Woo et al., 2023), (iii) 본 연구의 복원 디코더 토큰. 제안한 복원 디코더 방식이 모든 지표·모달리티 구성에서 가장 우수했고, 특히 단일 모달리티 시나리오에서 masked / memory 대비 ACC 가 각각 4.94%, 2.29% 향상되었다.

**복원 디코더의 차원과 깊이.**  Table 6 에 따르면 차원과 깊이를 늘릴수록 모든 평가 지표가 체계적으로 개선되었다. 특히 hidden dim 512 / depth 8 조합이 대부분의 모달리티 조합에서 최고 성능이었다. 즉, 고용량 복원 디코더가 결측 모달리티 정보 복원에 더 효과적이다.

---

## 5. 논의 및 결론 (Discussion and Conclusion)

본 연구는 임상 현장에서 빈번히 발생하는 결측 모달리티 문제를 다루기 위해 생리신호 특화 프레임워크 PhysioME 를 제안했다. PhysioME 는 MultiMAE (Bachmann et al., 2022) 기반 SSL 프레임워크인 SynthSleepNet (Lee et al., 2025) 을 토대로, 생리신호의 시간 특성과 센서 간 이질성을 모두 포착해 다양한 결측 시나리오를 강건하게 처리하도록 설계되었다.

이러한 구조적 설계 덕분에 PhysioME 는 다음 핵심 요소를 통해 강한 예측 성능과 일반화 능력을 보였다.

- **DP-NeuroNet**: 가중치 공유 dual-path 네트워크에 두 가지 시간 증강을 적용해 생리신호의 시간 동역학을 효과적으로 포착하고, 모달리티별 표현의 품질을 크게 끌어올린다. 이 고차원 표현이 예측 정확도와 일반화 능력 향상에 결정적이다 (Table 3).
- **복원 디코더**: 평균 대치(mean imputation)나 입력 복제와 같은 단순 imputation 대신, 모달리티 간 상호작용을 모델링한 의미론적 복원을 수행한다. stop-gradient 로 다중모달 인코더로의 역전파를 차단하면서도 깊은 구조를 가져, 복원 태스크에 집중적이고 효과적인 학습이 가능하다 (Table 5·6).

임상 현장에서는 모든 생리신호를 항상 수집하기 어렵다. 뇌수술 중에는 EEG 측정이 물리적으로 제약될 수 있고 (Bitar et al., 2024), 항응고제 치료를 받거나 응고 기능 저하가 있는 환자에게는 ABP 측정이 부적절할 수 있다 (Puckett et al., 2003). 마찬가지로 심한 움직임 아티팩트는 ECG 신뢰성을 떨어뜨리고, 저체온 또는 말초 혈관수축 상태에서는 PPG 가 부정확해질 수 있다 (Littmann, 2021). 이러한 제약에도 불구하고 PhysioME 는 예측 안정성과 강건성을 일관되게 유지하여, 실제 의료 환경에서 임상 의사결정을 신뢰성 있게 보조할 수 있는 잠재력을 보여주었다.

다만 본 연구에는 몇 가지 한계가 있다. 첫째, 사전에 정해진 모달리티 조합 하에서만 평가가 이루어져, 완전히 새로운 조합이나 도메인 전이 시나리오에 대한 일반화는 추가 검증이 필요하다. 둘째, DP-NeuroNet 과 복원 디코더 구조가 학습 효율 면에서 다소 복잡하며, 새로운 모달리티마다 별도의 복원 디코더를 인스턴스화해야 한다는 확장성 측면의 과제가 있다. 향후 연구에서는 다기관 다중 데이터셋을 활용한 hospital-cross 일반화 평가, 그리고 결측뿐 아니라 임상 데이터에서 흔히 관찰되는 노이즈·아티팩트에 대한 강건성도 함께 검증할 계획이다.

---

## 부록 (Appendix)

### A. 배경: NeuroNet

NeuroNet (Lee et al., 2024) 은 대조학습과 마스크 예측을 결합한 SSL 프레임워크이다. 본 연구에서는 NeuroNet 을 핵심 구성요소로 채택하여 각 생리신호의 모달리티별 특징을 추출했다.

NeuroNet 인코더의 중심은 frame network 이며, 시계열 처리를 위해 하나의 공유 컨볼루션 블록과 커널 크기 3, 5, 7 의 세 병렬 특징 추출기로 구성된다 (각 추출기는 컨볼루션 residual block 구조). 출력 특징 벡터들은 결합된 후 MLP 를 거쳐 frame-wise 벡터를 만든다.

MAE 전략에 기반한 마스크 예측 태스크에서는 frame-wise 벡터의 일부를 무작위로 선택해 ViT 인코더에 class token 과 함께 입력한다. 인코더는 관측된 frame 으로부터 잠재 표현을 만들고, 디코더는 이 잠재 표현과 mask token 을 이용해 마스크된 frame 을 복원한다. 복원 손실은 마스크된 frame 위에서만 MSE 로 계산한다.

$$
\mathcal{L}_{\text{inter recon}} = \frac{1}{|\mathcal{M}|}\sum_{m \in \mathcal{M}}\bigl\|r^{(m)} - z^{(m)}\bigr\|_2^2
$$

여기서 $\mathcal{M}$ 은 마스크된 frame 인덱스 집합이며, $r$ 과 $z$ 는 각각 디코더와 frame network 출력이다.

또한 NeuroNet 은 NT-Xent 손실 기반 대조학습을 통해 동일 입력의 서로 다른 뷰 간 일관성을 보장한다. 한 입력 샘플로부터 두 개의 무작위 샘플을 만들고, 각각을 인코더와 MLP 에 통과시켜 잠재 벡터 $z_i, z_j$ 를 얻는다.

$$
\mathcal{L}_{\text{NT-Xent}} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i, z_k)/\tau)}
$$

NeuroNet 은 같은 인코더·디코더로 두 개의 독립 마스크 예측 태스크를 수행하므로, 총 손실은 두 복원 손실과 한 대조 손실의 가중합이다.

$$
\mathcal{L}_{\text{total}} = \frac{1}{2}\bigl(\mathcal{L}_{\text{inter recon 1}} + \mathcal{L}_{\text{inter recon 2}}\bigr) + \lambda\,\mathcal{L}_{\text{NT-Xent}}
$$

여기서 $\lambda$ 는 복원 손실과 대조 손실 사이의 균형을 조절하는 하이퍼파라미터이다.

### B. 하이퍼파라미터 설정

실험 환경: Intel Xeon Gold 6338 CPU (2.00 GHz), 256 GB RAM, NVIDIA RTX 6000 Ada × 2. 데이터 전처리 및 모델 구현은 PyTorch 2.7.1 / Python 3.11 기반.

| 항목 | DP-NeuroNet (Sleep / Hypotension) | PhysioME (Sleep / Hypotension) | Downstream (Sleep / Hypotension) |
|---|---|---|---|
| epoch | 50 / 100 | 50 / 50 | 20 / 20 |
| batch size | 128 / 128 | 512 / 512 | 512 / 512 |
| frame size (sec) | 3 / 3 | — | — |
| overlap step (sec) | 4 / 3 | — | — |
| sampling rate | 100 / 128 | — | — |
| encoder dim | 512 / 512 | 512 / 512 | — |
| encoder depth | 8 / 8 | 6 / 8 | — |
| encoder head | 6 / 6 | 8 / 8 | — |
| decoder dim | 256 / 256 | 256 / 256 | — |
| decoder depth | 8 / 8 | 4 / 4 | — |
| decoder head | 4 / 4 | 8 / 8 | — |
| recon decoder dim | — | 256 / 256 | — |
| recon decoder depth | — | 8 / 8 | — |
| recon decoder head | — | 8 / 8 | — |
| projection hidden | [1024, 512] / [1024, 512] | [1024, 512] / [1024, 512] | — |
| mask ratio | 0.8 / 0.8 | 0.4 / 0.4 | — |
| balance scale | 1.0 / 1.0 | $\alpha=\beta=\gamma=1.0$ | — |
| temperature τ | — | 0.05 / 0.05 | — |
| LoRA r / α / dropout | — | 16 / 16 / 0.05 | — |
| optimizer | AdamW | AdamW | AdamW |
| optimizer momentum | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) |
| learning rate | 1e-5 / 1e-5 | 2e-4 / 2e-4 | 1e-5 / 1e-5 |

(Appendix Table 7 재구성. 원문 PDF 의 표가 일부 정렬 깨짐 — 값은 PDF 텍스트 추출본 기준이며, 정확한 수치는 원문 Table 7 을 직접 확인 권장.)

---

## 참고문헌 (References)

1. Bachmann, R.; Mizrahi, D.; Atanov, A.; Zamir, A. (2022). *MultiMAE: Multi-modal Multi-task Masked Autoencoders.* ECCV, 348–367. Springer.
2. Bitar, R.; Khan, U. M.; Rosenthal, E. S. (2024). *Utility and rationale for continuous EEG monitoring: a primer for the general intensivist.* Critical Care, 28(1): 244.
3. Deldari, S.; Spathis, D.; Malekzadeh, M.; Kawsar, F.; Salim, F. D.; Mathur, A. (2024). *CroSSL: Cross-modal self-supervised learning for time-series through latent masking.* Proc. WSDM, 152–160.
4. Esteva, A.; Robicquet, A.; Ramsundar, B.; et al. (2019). *A guide to deep learning in healthcare.* Nature Medicine, 25(1): 24–29.
5. Hu, E. J.; Shen, Y.; Wallis, P.; et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
6. Iranfar, A.; Arza, A.; Atienza, D. (2021). *ReLearn: A robust machine learning framework in presence of missing data for multimodal stress detection from physiological signals.* IEEE EMBC, 535–541.
7. Kemp, B.; Zwinderman, A. H.; Tuk, B.; Kamphuisen, H. A.; Oberye, J. J. (2000). *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.* IEEE Trans. Biomed. Eng., 47(9): 1185–1194.
8. Lee, C.-H.; Kim, H.; Han, H.-j.; Jung, M.-K.; Yoon, B. C.; Kim, D.-J. (2024). *NeuroNet: A novel hybrid self-supervised learning framework for sleep stage classification using single-channel EEG.* arXiv:2404.17585.
9. Lee, C.-H.; Kim, H.; Yoon, B. C.; Kim, D.-J. (2025). *Toward Foundational Model for Sleep Analysis Using a Multimodal Hybrid Self-Supervised Learning Framework.* arXiv:2502.17481.
10. Lee, H.-C.; Park, Y.; Yoon, S. B.; Yang, S. M.; Park, D.; Jung, C.-W. (2022). *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.* Scientific Data, 9(1): 279.
11. Lee, J.; Kim, D.-S. (2023). *RobustSsF: Robust Missing Modality Brain Tumor Segmentation with Self-supervised Learning-Based Scenario-Specific Fusion.* ML4MHD Workshop, 43–53. Springer.
12. Littmann, L. (2021). *Electrocardiographic artifact.* Journal of Electrocardiology, 64: 23–29.
13. Puckett, L. G.; Barrett, G.; Kouzoudis, D.; Grimes, C.; Bachas, L. G. (2003). *Monitoring blood coagulation with magnetoelastic sensors.* Biosensors and Bioelectronics, 18(5–6): 675–681.
14. Reza, M. K.; Prater-Bennette, A.; Asif, M. S. (2024). *Robust multimodal learning with missing modalities via parameter-efficient adaptation.* IEEE TPAMI.
15. Sohn, K. (2016). *Improved deep metric learning with multi-class N-pair loss objective.* NeurIPS, 29.
16. Wang, S.; Cao, J.; Philip, S. Y. (2020). *Deep learning for spatio-temporal data mining: A survey.* IEEE TKDE, 34(8): 3681–3700.
17. Woo, S.; Lee, S.; Park, Y.; Nugroho, M. A.; Kim, C. (2023). *Towards good practices for missing modality robust action recognition.* AAAI, vol. 37, 2776–2784.
18. Wu, R.; Wang, H.; Chen, H.-T.; Carneiro, G. (2024). *Deep multimodal learning with missing modality: A survey.* arXiv:2409.07825.
19. Yang, C.; Xiao, D.; Westover, M. B.; Sun, J. (2021). *Self-supervised EEG representation learning for automatic sleep staging.* arXiv:2110.15278.
20. Zhao, Z.; Li, J.; Wang, L.; Wang, Y.; Lu, H. (2024). *MaskMentor: Unlocking the Potential of Masked Self-Teaching for Missing Modality RGB-D Semantic Segmentation.* ACM MM, 1915–1923.

---

> 본 노트는 arXiv:2510.11110v1 PDF 의 한국어 번역본이다. 본문의 표·그림 위치는 원문 PDF 를 함께 참고하라. 원본 PDF 경로: `obsidian_physiome_hetero/2510.11110v1.pdf`.
