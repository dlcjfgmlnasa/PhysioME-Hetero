---
tags: [hub, paper]
---

# 90 Paper — Hub

> 논문 작성 인덱스. 현재 작성 중인 PhysioME-Hetero (JBHI) 와 baseline 인용 자료 (PhysioME 원본 arXiv:2510.11110).

## 현재 작성 중

[[PhysioME-Hetero/MOC|→ PhysioME-Hetero MOC]]

| 섹션 | 노트 |
|---|---|
| 핵심 메시지 | [[PhysioME-Hetero/00. 핵심 메시지]] |
| Abstract | [[PhysioME-Hetero/01. Abstract 초안]] |
| Introduction | [[PhysioME-Hetero/02. Introduction 개요]] |
| Method — Hetero-bucket | [[PhysioME-Hetero/03. Method - Hetero-bucket]] |
| Method — Availability-aware | [[PhysioME-Hetero/04. Method - Availability-aware loss]] |
| Method — Presence embedding | [[PhysioME-Hetero/05. Method - Presence embedding]] |
| Experiments 설계 | [[PhysioME-Hetero/06. Experiments 설계]] |
| Ablation 3종 | [[PhysioME-Hetero/07. Ablation 3종]] |
| External transfer | [[PhysioME-Hetero/08. External transfer]] |
| JBHI 체크리스트 | [[PhysioME-Hetero/09. JBHI 체크리스트]] |
| 실험 실행 가이드 | [[PhysioME-Hetero/10. 실험 실행 가이드]] |
| Phase-1 TF-C | [[PhysioME-Hetero/11. Phase-1 TF-C]] |

## Baseline (인용)

[[PhysioME-Original/MOC|→ PhysioME 원본 MOC]]

`Lee et al., PhysioME: A Robust Multimodal Self-Supervised Framework for Physiological Signals with Missing Modalities, arXiv:2510.11110, 2025`

원본 paper의 섹션별 본문 노트:
- [[PhysioME-Original/00. 메타데이터]]
- [[PhysioME-Original/01. Abstract]]
- [[PhysioME-Original/02. Introduction]]
- [[PhysioME-Original/03. Related Work]]
- [[PhysioME-Original/04. Method]]
- [[PhysioME-Original/05. Experiments]]
- [[PhysioME-Original/06. Discussion]]

## Submission target

- **Primary**: IEEE JBHI (IF ~7.7) — [[PhysioME-Hetero/09. JBHI 체크리스트]]
- **Fallback**: npj Digital Medicine
- **Stretch**: ICLR 2027 (A2 gap이 dramatic할 경우)

## 작성 워크플로우

1. **Method draft** — 03/04/05 노트 정리 후 LaTeX section 작성
2. **Experiments** — 실험 결과 들어오면 06 노트 갱신 → table/figure 결정
3. **Ablation** — 07 노트의 A1/A2/A3 결과 들어오면 figure 만들기 (A2가 main figure)
4. **Discussion** — calibration + 실패 케이스 분석 추가
5. **Submission checklist** — 09 노트 따라 IEEE JBHI 제출 형식 점검

## 외부 참조

- 원본 PhysioME GitHub: `https://github.com/dlcjfgmlnasa/PhysioME`
- PhysioME-Hetero GitHub: `https://github.com/dlcjfgmlnasa/PhysioME-Hetero`
- 인용 (BibTeX 만들 시 필요):

```bibtex
@article{lee2025physiome,
  title={PhysioME: A Robust Multimodal Self-Supervised Framework for Physiological Signals with Missing Modalities},
  author={Lee, Cheol-Hui and ...},
  journal={arXiv:2510.11110},
  year={2025}
}
```

## 원본

- Drafts: `notes/90_Paper/PhysioME-Hetero/*.md`
- Baseline: `notes/90_Paper/PhysioME-Original/*.md`
