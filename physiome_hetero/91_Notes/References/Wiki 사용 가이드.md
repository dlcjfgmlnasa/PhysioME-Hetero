---
tags: [guide, meta]
created: 2026-05-04
---

# 🛠️ Wiki 사용 가이드

## 폴더 구조

```
physiome_hetero/                    ← Obsidian vault root 후보
├── 00_MOC.md                       ← 진입점 (reading paths + hub 인덱스)
├── 01_Overview/                    ← Project_Overview, Master_Plan (SSOT)
├── 02_Architecture/                ← Architecture_Hub + Decision_*.md 7종
├── 03_Pretraining/                 ← Phase1 / Phase2 / Configs + Hub
├── 04_Data/                        ← VitalDB / Hetero_NPZ / MIMIC3 + Hub
├── 05_Downstream/                  ← IOH / AKI / Mortality / Ablation / Calibration + Hub
├── 90_Paper/                       ← 논문 drafts + 인용 baseline
│   ├── PhysioME-Hetero/            ← 현재 작성 중 섹션 노트
│   └── PhysioME-Original/          ← arXiv:2510.11110 baseline
└── 91_Notes/                       ← Status timeline + 저자 메모
    ├── Status_*.md
    └── References/
```

## Obsidian 설정 권장

1. **Vault 루트**: `C:\Projects\PhysioME\physiome_hetero` (또는 `C:\Projects\PhysioME` 전체)
   - `physiome_hetero/`만 vault 루트로 잡으면 코드와 분리되어 깔끔
   - 전체 vault로 잡으면 `[[../models/physiome/model.py]]` 같은 코드 링크 가능
2. **Plugin 추천**:
   - **Templater** — frontmatter 자동화
   - **Dataview** — 표 자동 집계 (예: 모든 method 섹션 한 번에 표시)
   - **Mermaid** — 아키텍처 다이어그램 (`​```mermaid` 블록 지원)
3. **글꼴 / 한글**: 기본 설정으로 한글 잘 보임. CJK 폭 이슈는 ``Settings → Editor → Readable line length`` 끄면 개선

## 표기 규약

- 본문은 한글
- 코드/식별자/논문 제목/저자 표기는 영문 그대로
- 수식: LaTeX (Obsidian이 KaTeX 렌더링)
- 인용/참고: `[Lee et al. 2025]` 식 짧은 표기 + MOC에 출처 정리
- TODO 표시: `> [!todo]` 또는 `#todo`

## 주요 wiki-link 패턴

- 같은 폴더 내: `[[Decision_Hetero_Bucket]]`
- 다른 카테고리: `[[../02_Architecture/Decision_Hetero_Bucket]]`
- Hub 진입: `[[../02_Architecture/Architecture_Hub]]`
- 코드 파일 링크: `[[../../models/physiome/model.py]]` (vault root가 physiome_hetero일 때)
- 별칭: `[[Decision_Hetero_Bucket|hetero-bucket 결정]]` → "hetero-bucket 결정"으로 표시

## 본문 paste 워크플로우

1. 원본 LaTeX/Word에서 섹션 본문 복사
2. 해당 stub 파일 (예: `02. Introduction.md`) 열어 stub 표시 아래에 붙여넣기
3. 영문은 그대로 두거나 ChatGPT/DeepL로 한글 번역 후 paste
4. 번역 시 다음 용어는 영문 유지 권장:
   - SSL, MAE, NTXent, ViT, LoRA, MAP, IOH, AUPRC, AUROC
   - 모델 이름 (PhysioME, NeuroNet, BiosignalFM 등)
   - 데이터셋명 (VitalDB, MIMIC-III WDB)
