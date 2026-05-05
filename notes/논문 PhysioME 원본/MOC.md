---
tags: [moc, paper, physiome-original]
created: 2026-05-04
---

# 📄 PhysioME (원본) — 한글판 허브

> [!info]
> arXiv:2510.11110의 한글 번역 + 메타. 본문은 일부 stub 상태이며 사용자분이 원본 LaTeX/Word에서 paste하면 채워집니다 ([[../참고/저자 메모|저자 메모]] 참조).

> [!warning] 범위 메모 (2026-05-04)
> 원본 논문은 Sleep-EDFx와 VitalDB 두 데이터셋을 다뤘으나, **본 후속 연구는 VitalDB만** 다룹니다. 원본 논문 노트의 Sleep-EDFx 관련 기술은 정리 시 제거되었습니다 — 본문 paste 시점에 Sleep-EDFx 부분은 생략하거나 별도 보관 권장.

## 빠른 참조

- **제목**: PhysioME: A Robust Multimodal Self-Supervised Framework for Physiological Signals with Missing Modalities
- **저자**: Cheol-Hui Lee, Hwa-Yeon Lee, Min-Kyung Jung, Dong-Joo Kim
- **출처**: arXiv:2510.11110, 2025
- **GitHub**: ``https://github.com/dlcjfgmlnasa/PhysioME``

## 섹션

- [[00. 메타데이터]]
- [[01. Abstract]]
- [[02. Introduction]] — *본문 paste 필요*
- [[03. Related Work]] — *본문 paste 필요*
- [[04. Method]] — *코드 기반 부분 정리됨, 본문 검수 필요*
- [[05. Experiments]] — *본문 paste 필요*
- [[06. Discussion]] — *본문 paste 필요*

## 신규 논문에서의 위치

PhysioME 원본은 **PhysioME-Hetero의 baseline + 출발점**:
- baseline narrative: [[../논문 PhysioME-Hetero/00. 핵심 메시지]]
- 직접 비교 대상 (A1 ablation): [[../논문 PhysioME-Hetero/07. Ablation 3종]]

## 우리가 발견한 원본의 한계 (신규 논문에서 정조준)

1. **Synthetic missing modality 가정** — complete-modality 데이터 분포에서 학습 후 인위적 drop으로 시뮬레이션
2. **모든 modality 보유 case만 사용** — VitalDB의 데이터 절반 이상 잠재적 손실
3. **(2개 코드 버그)** — `forward_restoration_decoder`의 split_size 불일치, decoder 출력의 잘못된 ``.detach()`` 위치 → restoration / inter-recon loss가 사실상 decoder 학습 신호를 제공하지 않음
   - 자세한 내용: [[../논문 PhysioME-Hetero/03. Method - Hetero-bucket]]
