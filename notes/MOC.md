---
tags: [moc, paper, physiome]
created: 2026-05-04
---

# 📚 PhysioME 연구 — 작업 허브 (MOC)

> [!info]
> 이 vault는 두 개의 논문을 추적합니다 — **이미 발표된 PhysioME 원본**과 **현재 작업 중인 PhysioME-Hetero 후속작**. 좌측의 폴더 구조 + 아래 링크로 탐색하세요.

## 📄 논문 1 — PhysioME (원본, arXiv:2510.11110)

이미 발표된 작업. **PhysioME-Hetero의 baseline + narrative 출발점**.

- [[논문 PhysioME 원본/MOC|→ PhysioME 원본 허브로]]
- 핵심 인용: ``Lee et al., PhysioME: A Robust Multimodal Self-Supervised Framework for Physiological Signals with Missing Modalities, arXiv:2510.11110, 2025``

## 📄 논문 2 — PhysioME-Hetero (신규, **타깃: IEEE JBHI**)

현재 작성 중. **"Synthetic missing → Real heterogeneous missing" distribution-shift 프레이밍**으로 PhysioME 핵심 한계 정조준.

- [[논문 PhysioME-Hetero/MOC|→ PhysioME-Hetero 허브로]]
- 핵심 contribution 요약: [[논문 PhysioME-Hetero/00. 핵심 메시지]]

## 🔗 참고

- [[참고/Wiki 사용 가이드]]
- [[참고/저자 메모]] — 본문 paste 필요한 섹션 추적
- 원본 PhysioME GitHub: ``https://github.com/dlcjfgmlnasa/PhysioME``
- RL-BioAug GitHub (이전 작업, idea 출처): ``https://github.com/dlcjfgmlnasa/RL-BioAug``
- Biosignal-Foundation-Model GitHub (별도 진행 프로젝트, 인프라 일부 차용): ``https://github.com/dlcjfgmlnasa/Biosignal-Foundation-Model``

## 🧭 작업 진행 상태 (2026-05-04 기준)

- ✅ 코드 인프라 (Tasks 1–7) 완료
- ✅ Biosignal-Foundation-Model에서 quality checks / MIMIC parser / IOH downstream 통합 완료 (Tasks 14–18)
- ⏳ Task 8 — 실제 VitalDB pretraining 1차 run (사용자 GPU)
- ⏳ Tasks 9–13 — downstream 평가 + ablation + transfer + paper draft

## 📌 빠른 결정 인덱스

| 질문 | 답 | 근거 |
|---|---|---|
| 데이터셋 | **VitalDB only** + MIMIC-III WDB transfer 1회 | [[논문 PhysioME-Hetero/06. Experiments 설계]] |
| 타깃 venue | **IEEE JBHI** (1순위), npj Digital Medicine (대안) | [[논문 PhysioME-Hetero/09. JBHI 체크리스트]] |
| 모달리티 | ABP, ECG, PPG (3종) | [[논문 PhysioME-Hetero/06. Experiments 설계]] |
| 핵심 contribution | Hetero-bucket + Availability-aware loss + 3-state presence embedding | [[논문 PhysioME-Hetero/00. 핵심 메시지]] |
