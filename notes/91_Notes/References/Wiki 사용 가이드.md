---
tags: [guide, meta]
created: 2026-05-04
---

# 🛠️ Wiki 사용 가이드

## 폴더 구조

```
notes/
├── MOC.md                          ← 작업 허브 (홈)
├── 논문 PhysioME 원본/             ← 발표된 논문 한글 번역 + 메타
├── 논문 PhysioME-Hetero/           ← 신규 논문 아웃라인 + 초안
└── 참고/                            ← 가이드, 저자 메모, 외부 reference
```

## Obsidian 설정 권장

1. **Vault 루트**: `C:\Projects\PhysioME` (이 워크스페이스 전체)
   - `.gitignore`에 `notes/`가 빠져있으니 git에 같이 commit 가능
   - 코드와 노트가 같은 vault 안에 있어 `[[code 파일]]` 링크 가능
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

- 같은 폴더 내: `[[01. Abstract]]`
- 다른 폴더: `[[논문 PhysioME-Hetero/MOC]]`
- 코드 파일 링크: `[[../models/physiome/model.py]]` (Obsidian에서 클릭 시 외부 에디터로)
- 별칭: `[[01. Abstract|초록]]` → "초록"으로 표시

## 본문 paste 워크플로우

1. 원본 LaTeX/Word에서 섹션 본문 복사
2. 해당 stub 파일 (예: `02. Introduction.md`) 열어 stub 표시 아래에 붙여넣기
3. 영문은 그대로 두거나 ChatGPT/DeepL로 한글 번역 후 paste
4. 번역 시 다음 용어는 영문 유지 권장:
   - SSL, MAE, NTXent, ViT, LoRA, MAP, IOH, AUPRC, AUROC
   - 모델 이름 (PhysioME, NeuroNet, BiosignalFM 등)
   - 데이터셋명 (VitalDB, MIMIC-III WDB)
