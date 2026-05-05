---
tags: [moc, paper, physiome-hetero]
created: 2026-05-04
target_venue: IEEE JBHI
status: outlining
---

# 📄 PhysioME-Hetero — 신규 논문 작업 허브

> [!info]
> **타깃 venue**: IEEE Journal of Biomedical and Health Informatics (IF ~7.7)
> **2순위**: npj Digital Medicine | **stretch**: ICLR 2027

## 한 줄 요약

[[00. 핵심 메시지|"Synthetic missing modality는 OOD 일반화 문제다 — real heterogeneous 분포로 학습하면 사라진다"]]

## 섹션 인덱스

### 0. 메시지·서지

- [[00. 핵심 메시지]]
- [[01. Abstract 초안]]
- [[02. Introduction 개요]]

### 3–5. Method

- [[03. Method - Hetero-bucket]]
- [[04. Method - Availability-aware loss]]
- [[05. Method - Presence embedding]]

### 6–8. Experiments

- [[06. Experiments 설계]]
- [[07. Ablation 3종]]
- [[08. External transfer]]

### 9–11. Submission · Runbook · Phase-1 SSL

- [[09. JBHI 체크리스트]]
- [[10. 실험 실행 가이드]]
- [[11. Phase-1 TF-C]]

## 비교 — 원본 PhysioME vs PhysioME-Hetero

| 차원 | PhysioME 원본 | PhysioME-Hetero (ours) |
|---|---|---|
| **학습 데이터** | Complete-modality 만 | **Real heterogeneous 분포** (bucket sampling) |
| **결손 학습 방식** | Synthetic drop 시뮬레이션 | Real missing + synth drop **분리** |
| **모달리티 상태 표현** | (없음) | **3-state presence embedding** |
| **Loss 구성** | Inter-recon + miss-recon + contra (모두 모든 batch에서) | **Availability-aware 분해** (loss별 적용 조건 분리) |
| **Restoration loss 적용** | 모든 batch (synth drop) | Complete bucket 또는 synth-drop된 모달만 |
| **인코더 backbone** | timm ViT Block (LayerNorm + MHA + GELU FFN) + sinusoidal pos_embed + interpolation | **uni2ts-derived TransformerEncoder** (RMSNorm + GQA + GLU FFN + RoPE) — 임의 길이 네이티브 지원. **NeuroNet 인코더 + PhysioME 멀티모달 인코더 + MAE 디코더 + restoration 디코더 모두 통일** |
| **Phase-1 SSL loss** | SimCLR × 2 (NT-Xent), sleep-EEG augmentation (segment crop · permutation) | **Time-Frequency Consistency** ([[11. Phase-1 TF-C]]) — L_T + L_F + L_TF + recon, augmentation 미사용 |
| **버그** | split_size 불일치, decoder ``.detach()`` | 둘 다 fix |

## 작업 순서 (실험 기준)

1. ⏳ Task 8 — VitalDB pretraining 본격 run
2. ⏳ Task 9 — IOH downstream + 보조 task 2개 평가
3. ⏳ Task 10 — Ablation A1/A2/A3 ([[07. Ablation 3종]])
4. ⏳ Task 11 — MIMIC-III WDB external transfer ([[08. External transfer]])
5. ⏳ Task 12 — Calibration + 실패 케이스 정성 분석
6. ⏳ Task 13 — Paper draft 정식 작성
