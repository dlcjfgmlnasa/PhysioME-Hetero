# Plan: Evaluation — PhysioME-Hetero

> 본 파일은 **estimator** 에이전트의 평가 기준입니다. 항상 이 파일을 먼저 읽고 평가를 수행하세요.
> Single source of truth: `obsidian_physiome_hetero/00_MOC.md` 와 `obsidian_physiome_hetero/05_Downstream/Ablations_A1_A2_A3.md`, `obsidian_physiome_hetero/04_Data/MIMIC3_WDB_Transfer.md`, `obsidian_physiome_hetero/90_Paper/PhysioME-Hetero/09. JBHI 체크리스트.md`.

---

## 1. 평가 목표 / Target Venue

- **Primary venue**: IEEE JBHI (Journal of Biomedical and Health Informatics, IF ~7.7)
- **Fallback**: npj Digital Medicine
- **Stretch**: ICLR 2027 (real vs synthetic OOD gap이 dramatic할 경우)

---

## 2. 핵심 메시지 (one-line)

> Multimodal SSL with synthetic missing-modality drop is an out-of-distribution generalization problem. We close the gap by training on the real heterogeneous availability distribution itself.

평가는 이 메시지를 **반박불가하게** 입증해야 합니다.

---

## 3. 평가 컴포넌트

### 3.1 Downstream Tasks
| Task | 데이터 | 라벨 | 메트릭 | 코드 |
|---|---|---|---|---|
| **IOH** (primary) | VitalDB | MAP<65 sustained ≥1min, 5min horizon | AUROC / AUPRC / Sens@Sp90 | `downstream/run_ioh.py` |
| **AKI** (secondary) | VitalDB | KDIGO Cr-based, postop 7d | AUROC / AUPRC | `downstream/run_aki.py` |
| **Mortality** (transfer) | MIMIC-III WDB | hospital_expire_flag | AUROC, 5-fold CV | `downstream/run_mortality.py` |

### 3.2 Ablation 3종 (must-haves)
| ID | 비교 | 목적 | 코드 |
|---|---|---|---|
| **A1** | Synth-only baseline (PhysioME v1) vs Hetero-train (ours), matched compute | Hetero-train의 순수 효과 | 두 ckpt를 동일 평가 파이프라인에서 비교 |
| **A2** | Test set을 real-missing vs synth-missing으로 분할 → ours가 real-missing에서 우세 | OOD-attack counter (가장 중요) | `downstream/run_ablation_a2.py` |
| **A3** | restoration_only_on_complete: true vs false | Loss decomposition이 restoration 성능을 해치지 않는지 | `config/vital_db/physiome_hetero_a3.yaml` 로 train한 ckpt와 비교 |

### 3.3 Calibration (JBHI 가산점)
- ECE (Expected Calibration Error) + reliability diagram
- 코드: `downstream/calibration.py` (saved preds 받아서 plot)
- 운영점 (Sens@Sp90, threshold) 명시적 보고

### 3.4 7-Subset Modality Robustness
- 모든 downstream 평가는 **7개 modality subset** ({ABP}, {ECG}, {PPG}, {ABP,ECG}, {ABP,PPG}, {ECG,PPG}, {ABP,ECG,PPG}) 모두에 대해 수행
- `downstream/run_ioh.py` 등이 이미 7-subset sweep 지원
- 보고 형식: 7행 표 (subset × metric) + 마지막 행에 평균

---

## 4. 평가 컨트랙트

| 인터페이스 | 형태 | 비고 |
|---|---|---|
| 평가 입력 | `ckpt_path` + `data_root` + `output_dir` | hetero ckpt 포맷 (model_state + modality_backbone_param + ch_names) |
| 평가 출력 | `results.json` (per-subset metrics) + `preds.npz` (--save_preds 시) + `calibration.png` (옵션) | preds.npz로 calibration 후처리 |
| 시드 | 최소 3개 (예: 42/123/2024) — mean ± std 보고 | JBHI reviewer가 단일 시드 결과를 의심 |
| 환자/케이스 분할 | subject-level (case_id 기준), test leak 방지 | linear_probe든 zero_shot이든 동일 |

---

## 5. 평가 기준 (체크리스트)

### 5.1 코드 정확성
- [ ] 평가 스크립트가 실제로 ckpt를 로드하고 forward에 nan/inf 없는지 확인
- [ ] 7-subset enumeration이 정확히 7개 (`itertools.combinations` 으로 1~3개 부분집합)
- [ ] zero-fill된 absent modality에 presence_mask가 정확히 전달되는지
- [ ] AUPRC 계산 시 class imbalance (IOH는 positive ratio ~10%) 고려 — `average='binary'` 또는 명시적 positive class

### 5.2 결과 신뢰성
- [ ] 시드 3개 이상 평균
- [ ] subject-level split (case leakage 없음)
- [ ] confidence interval (bootstrap N=1000) 또는 std 명시
- [ ] A1 비교는 동일 Phase-1 ckpt로 시작 (Phase-1 randomness가 결과에 영향 X)

### 5.3 보고 품질
- [ ] 7-subset 별 metric 표 + 평균
- [ ] A2: real-missing vs synth-missing performance gap 명시 (figure 권장)
- [ ] Calibration plot (IOH primary)
- [ ] 운영점 (clinical threshold) 명시 — 단순 AUROC만 보고하지 말 것
- [ ] External transfer 결과 (MIMIC-III) 별도 표

---

## 6. 태스크 — 진행 상태

### 완료 (`[x]`)
- [x] **[High]** Downstream evaluator 인프라 (`run_ioh`, `run_aki`, `run_mortality`, `run_ablation_a2`, `calibration`)
- [x] **[High]** Smoke test 통과 (7-subset inference 검증)
- [x] **[High]** Linear-probe refactor (SVC → LR + sampled subsets, `pretrained/physiome/probe_utils.py`) — N≥4 modal 확장 선결조건 (Step 1)

### 진행 예정 (`[ ]`)

- [ ] **[High]** **IOH primary 평가 실행**
  - 입력: `ckpt/physiome_hetero/best.pt` + IOH downstream 데이터
  - 출력: `results/ioh/{subset}_{seed}.json` × 7 × 3 + 종합 표
  - 의존성: model-architect의 Phase-2 pretraining 완료
  - 참고: `--save_preds` 활성화하여 calibration plot 생성

- [ ] **[High]** **A1 ablation: Synth-only vs Hetero**
  - 입력: 두 ckpt (`ckpt/physiome_a1_synthonly/best.pt` + `ckpt/physiome_hetero/best.pt`)
  - 출력: `results/ablation_a1/comparison.json` (per-subset delta)
  - 의존성: A1 baseline pretraining 완료
  - 참고: matched compute 확인 (epoch×batch).

- [ ] **[High]** **A2 ablation: real-missing vs synth-missing**
  - 입력: `ckpt/physiome_hetero/best.pt` + downstream test set (real-missing flag 있는 케이스)
  - 출력: `results/ablation_a2/{real,synth}_missing.json`
  - 의존성: data-engineer의 hetero 데이터에 modality availability 메타데이터가 보존되어 있어야 함
  - 참고: **이게 paper의 핵심 figure** — gap이 명확히 보이면 ICLR도 노릴 수 있음

- [ ] **[Medium]** **A3 ablation: restoration_only_on_complete toggle**
  - 입력: 두 ckpt (`physiome_hetero/best.pt` + `physiome_a3_no_complete_only/best.pt`)
  - 출력: `results/ablation_a3/comparison.json`
  - 의존성: A3 pretraining 완료
  - 참고: A3가 **거의 비슷하면** decomposition이 restoration을 안 해친다는 증거.

- [ ] **[Medium]** **AKI 평가**
  - 입력: hetero ckpt + AKI 라벨 데이터
  - 출력: `results/aki/{subset}_{seed}.json`
  - 의존성: data-engineer의 AKI 라벨링 완료
  - 참고: AKI는 IOH보다 imbalanced함 — AUPRC가 main metric.

- [ ] **[Medium]** **External transfer (MIMIC-III mortality)**
  - 입력: hetero ckpt + MIMIC-III mortality 데이터
  - 출력: `results/mortality_transfer/{zero_shot,linear_probe}.json`
  - 의존성: data-engineer의 MIMIC-III mortality 데이터 매칭 완료
  - 참고: zero_shot 모드 = encoder freeze + 5-fold CV. linear_probe 모드 = 80/20 + LR train.

- [ ] **[Low]** **Calibration 후처리**
  - 입력: `--save_preds` 결과 npz
  - 출력: `results/calibration/{task}.png` + ECE 수치
  - 의존성: 모든 downstream 평가 완료
  - 참고: Temperature scaling 적용 전/후 비교 권장.

- [ ] **[Low]** **실패 케이스 정성 분석 (JBHI 가산점)**
  - 입력: saved preds + 원본 signal
  - 출력: 잘못 예측된 high-confidence 케이스의 signal plot + 임상적 해석
  - 의존성: 평가 결과 전체
  - 참고: 5~10 케이스 정도면 충분. clinically interpretable한 failure mode 1~2개 찾으면 paper에 넣을 가치 있음.

---

## 7. 평가 보고 형식

평가 완료 시 다음 구조로 markdown 보고:

```
## 📊 Estimator Report

### 평가 대상
- ckpt: <path>
- task: IOH / AKI / Mortality / A1 / A2 / A3
- seeds: [42, 123, 2024]

### 핵심 결과
| Subset | AUROC | AUPRC | Sens@Sp90 |
|---|---|---|---|
| {ABP} | ... | ... | ... |
| ... | ... | ... | ... |
| 평균 | ... | ... | ... |

### 발견된 이슈
- (Critical/Warning/Suggestion으로 분류)

### 다음 권장 액션
- ...
```

---

## 8. Error Handling

- ckpt가 존재하지 않으면 → 사용자에게 어떤 pretraining 단계가 빠졌는지 보고
- 데이터셋이 비어있으면 → data-engineer 호출 권장
- 평가 메트릭이 비정상값 (예: AUROC < 0.5) → 모델 forward에 nan/inf 있는지 확인하고 model-architect에게 디버깅 요청

---

## 9. 다음 액션

`@estimator`를 호출할 때 가장 먼저 처리할 태스크는 **IOH primary 평가** 입니다 (Section 6의 첫 미완료 항목). 단, **model-architect의 Phase-2 pretraining 완료가 선행 조건**입니다. ckpt 없으면 먼저 model-architect 호출.
