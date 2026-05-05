---
tags: [hub, notes]
---

# 91 Notes — Hub

> Status timeline + smoke test 결과 + 저자 메모.

## Status timeline

| 날짜 | 상태 노트 | 핵심 변화 |
|---|---|---|
| 2026-05-05 | [[Status_2026_05_05]] | freq_proj fix · wiki BFM 스타일로 재편 · plan 파일 작성 |

(이전 status는 git log + memory file 참조)

## 검증 결과

- Smoke tests 3종 (모두 CPU, 통과) — [[Status_2026_05_05#smoke 검증]]:
  - `experiments/smoke_test_neuronet_tfc.py` — Phase-1 TF-C 4 loss + gradient flow + 30 step convergence
  - `experiments/smoke_test_hetero.py` — Phase-2 7 bucket + presence states + availability-aware loss
  - `experiments/smoke_test_downstream.py` — frozen encoder + 7-subset inference

## 저자 메모

[[References/저자 메모]] — 본문 paste 필요한 섹션, TODO, 잡 메모

## Wiki 사용 가이드

[[References/Wiki 사용 가이드]] — 처음 vault 사용 시 참고

## 외부 메모리

이 wiki와 별개로 `C:\Users\SNUH_VitalLab_LEGION\.claude\projects\C--Projects-PhysioME\memory\` 에 auto-memory 시스템:
- `MEMORY.md` — 인덱스
- `project_physiome_state.md` — codebase 스냅샷 + extension levers
- `user_role.md` — 저자 정보

→ AI 대화 컨텍스트 유지용. wiki는 사람용 SSOT, memory는 AI 에이전트용 SSOT.

## 원본

- Status notes: `obsidian_physiome_hetero/91_Notes/Status_*.md`
- 저자 메모: `obsidian_physiome_hetero/91_Notes/References/`
