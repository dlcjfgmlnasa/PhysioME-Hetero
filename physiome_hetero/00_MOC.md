---
tags: [moc]
project: physiome-hetero
created: 2026-05-04
updated: 2026-05-05
---

# PhysioME-Hetero — Map of Content

> Wiki 진입점. **목적별 reading path**를 먼저 제시한 뒤 카테고리별 Hub를 연결한다. 각 노트 끝에는 원본 코드 파일 위치를 명시한다 (이 wiki는 코드 옆 .md를 수정하지 않는다).

---

## 🚪 처음 오신 분 (입문 경로)

순서대로 한 번씩 읽으면 1시간 내에 큰 그림이 잡힌다.

1. [[01_Overview/Project_Overview]] — 무엇을 만드는가, 한 페이지 큰 그림 ⭐
2. [[01_Overview/Master_Plan]] — 데이터/venue/모달리티 SSOT
3. [[02_Architecture/Architecture_Hub]] — 모델 스택과 핵심 설계 결정 7종
4. [[03_Pretraining/Pretraining_Hub]] → [[03_Pretraining/Phase1_NeuroNet_TFC]] → [[03_Pretraining/Phase2_PhysioME_Hetero]]
5. [[04_Data/Data_Hub]] — VitalDB SSL + downstream 데이터 파이프라인
6. [[05_Downstream/Downstream_Hub]] — IOH primary + AKI/Mortality + Ablation 3종

---

## 🎯 목적별 reading path

### "왜 이런 모델이 나왔는지 알고 싶다" — 설계 결정 narrative

[[02_Architecture/Decision_Hetero_Bucket]] → [[02_Architecture/Decision_Presence_Embedding]] → [[02_Architecture/Decision_BFM_Transformer_Port]] → [[02_Architecture/Decision_TFC_over_SimCLR]] → [[02_Architecture/Decision_HandRolled_LoRA]] → [[02_Architecture/Decision_FreqProj_Linear]]

### "Phase 1 학습을 돌리고 싶다"

[[03_Pretraining/Phase1_NeuroNet_TFC]] → [[03_Pretraining/Configs]] → [[04_Data/VitalDB_SSL]]

### "Phase 2 학습을 돌리고 싶다"

[[03_Pretraining/Phase1_NeuroNet_TFC]] (완료 가정) → [[03_Pretraining/Phase2_PhysioME_Hetero]] → [[02_Architecture/Decision_HandRolled_LoRA]]

### "내 데이터를 추가하고 싶다"

[[04_Data/Data_Hub]] → [[04_Data/Hetero_NPZ_Schema]] → [[04_Data/VitalDB_SSL]] (또는 [[04_Data/MIMIC3_WDB_Transfer]])

### "Downstream 평가를 붙이고 싶다"

[[05_Downstream/Downstream_Hub]] → [[05_Downstream/IOH_Primary]] → [[05_Downstream/Ablations_A1_A2_A3]]

### "최근 무슨 일이 있었나"

[[91_Notes/Notes_Hub]] (timeline) — 최신: [[91_Notes/Status_2026_05_05]]

### "논문(JBHI 1-pass) 작성에 필요한 자료를 찾는다"

- 현재 작성 중: [[90_Paper/PhysioME-Hetero/MOC]] — section별 draft
- 핵심 메시지: [[90_Paper/PhysioME-Hetero/00. 핵심 메시지]]
- Submission checklist: [[90_Paper/PhysioME-Hetero/09. JBHI 체크리스트]]
- 원본 baseline: [[90_Paper/PhysioME-Original/MOC]]

---

## 📚 카테고리별 Hub

| 카테고리 | Hub | 무엇을 다루나 |
|---|---|---|
| 01 Overview | [[01_Overview/Overview_Hub]] | 진입점·마스터플랜·SSOT |
| 02 Architecture | [[02_Architecture/Architecture_Hub]] | 모델 컴포넌트·설계 결정 7종 |
| 03 Pretraining | [[03_Pretraining/Pretraining_Hub]] | 2-Phase SSL·Loss·Config |
| 04 Data | [[04_Data/Data_Hub]] | 데이터 파이프라인·전처리·QC |
| 05 Downstream | [[05_Downstream/Downstream_Hub]] | IOH/AKI/Mortality·Ablation 3종·Calibration |
| 90 Paper | [[90_Paper/Paper_Hub]] | **현재 작성 본문** + 원본 baseline 참조 |
| 91 Notes (메타) | [[91_Notes/Notes_Hub]] | Status timeline·smoke 결과·저자 메모 |

---

## 📌 빠른 결정 인덱스

| 질문 | 답 | 근거 |
|---|---|---|
| 데이터셋 | **VitalDB only** + MIMIC-III WDB transfer 1회 | [[01_Overview/Master_Plan]] |
| 타깃 venue | **IEEE JBHI** (1순위) / npj Digital Medicine / ICLR 2027 | [[90_Paper/PhysioME-Hetero/09. JBHI 체크리스트]] |
| 모달리티 | ABP / ECG / PPG @ 100 Hz, 60 s windows | [[01_Overview/Master_Plan]] |
| Backbone | BFM-derived TransformerEncoder (RMSNorm + GQA + GLU FFN + RoPE) | [[02_Architecture/Decision_BFM_Transformer_Port]] |
| Phase-1 SSL | NeuroNet + **TF-C** (L_T + L_F + L_TF + recon) | [[02_Architecture/Decision_TFC_over_SimCLR]] |
| Phase-2 SSL | PhysioME hetero-bucket + availability-aware loss + 3-state presence | [[02_Architecture/Decision_Hetero_Bucket]] |
| LoRA | hand-rolled (`models/transformer/lora.py`), peft 제거 | [[02_Architecture/Decision_HandRolled_LoRA]] |
| 핵심 contribution | Hetero-bucket + Availability-aware + Presence embedding | [[90_Paper/PhysioME-Hetero/00. 핵심 메시지]] |

---

## 🧭 진행 상태 (2026-05-05)

- ✅ 코드 인프라 (Tasks 1–7) 완료
- ✅ BFM Transformer 포팅 (Task 21)
- ✅ Phase-1 SimCLR → TF-C 전환 (Task 22)
- ✅ PhysioME RoPE-everywhere (Task 23)
- ✅ peft → 자체 LoRA (Task 24)
- ✅ Freq view zero-pad → learned linear lift (`freq_proj`)
- ⏳ Task 8 — VitalDB pretraining 본격 run (사용자 GPU)
- ⏳ Tasks 9–13 — downstream 평가 + ablation + transfer + paper draft

---

## ⚠️ 본 wiki 사용 시 주의

- **원본 .md는 절대 수정하지 않음** — wiki는 요약·연결·진입점만 제공한다. 실제 진실은 원본 코드/논문 파일에 있고, 각 노트 끝의 "원본" 섹션이 가리킨다.
- **수치/설정은 drift 가능성**: `config/vital_db/*.yaml`을 항상 신뢰하라 — wiki 값은 작성 시점 스냅샷일 수 있다.
- **간헐적 모순**: 이전 plan이 새 결정으로 덮인 경우 [[91_Notes/Notes_Hub]] timeline의 최신 항목을 진실로 본다.
- **외부 referenced 코드**: `references/Biosignal-Foundation-Model/` 은 별도 프로젝트의 snapshot — 인용용일 뿐 import 금지.
