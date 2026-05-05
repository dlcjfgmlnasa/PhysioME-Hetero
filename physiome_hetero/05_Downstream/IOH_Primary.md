---
tags: [downstream, ioh, primary]
---

# IOH — Primary Downstream Task

## 임상 정의

**Intraoperative Hypotension (IOH)** — 수술 중 평균동맥압(MAP)이 임상적 기준 이하로 *지속*되는 상태.

본 연구의 라벨:
- **Threshold**: MAP < 65 mmHg
- **Duration**: ≥ 1 minute (sustained)
- **Horizon**: window 종료 후 5 min 이내 발생 여부

→ KDIGO-style sustained labeling (단발성 spike는 라벨 X).

## 왜 primary인가

- SNU Vital Lab의 가장 강한 임상 narrative — IOH 예측은 vital lab의 본업
- ABP가 라벨 자체의 source modality라서 **modality 결손 narrative와 자연스러운 결합** ("ABP가 없을 때도 IOH 예측 가능?")
- 임상 가치 명확 — IOH는 postop AKI/mortality와 인과 연결되어 reviewer가 metric을 이해하기 쉬움

## 라벨링 코드

`downstream/tasks/hypotension.py::HypotensionDataset`:
- BFM의 KDIGO-style sustained-IOH labeler를 surgical pull
- window × horizon sweep 가능 (parametrized)
- 기본: window=60s, horizon=5min

## Eval

`downstream/run_ioh.py`:

```bash
python -m downstream.run_ioh \
    --ckpt ckpt/physiome_hetero/best.pt \
    --data_root data/downstream/ioh/ \
    --output_dir results/ioh/ \
    --seeds 42 123 2024 \
    --save_preds  # for calibration analysis
```

Output:
```
results/ioh/
├── seed_42/
│   ├── ABP.json
│   ├── ECG.json
│   ├── PPG.json
│   ├── ABP+ECG.json
│   ├── ABP+PPG.json
│   ├── ECG+PPG.json
│   └── ABP+ECG+PPG.json
├── seed_123/...
├── seed_2024/...
└── summary.json   ← 7 × 3 × {AUROC, AUPRC, Sens@Sp90}
```

## Metrics

- **AUROC** — 표준
- **AUPRC** — IOH는 class imbalance (positive ~10%) 라 AUPRC가 main metric
- **Sens@Sp90** — clinical operating point (특정 specificity에서의 sensitivity)

## 7-subset performance gradient (예상)

```
{ABP}            ← 라벨 source. 고성능 예상.
{ABP, ...}       ← ABP 포함 시 차이 작음
{ECG, PPG}       ← ABP 없는 hardest case. 본 연구의 main claim 검증 지점.
{ECG} or {PPG}   ← single modal worst.
```

A2 ablation에서 *동일 7-subset 안에서* real-missing(라벨 case에 ABP 자체가 없음) vs synth-missing (ABP 있지만 평가에서 인위 drop) 의 gap을 측정.

## 결정

- [[../02_Architecture/Decision_Hetero_Bucket]] — 학습 분포 일치
- [[Ablations_A1_A2_A3]] — 평가 ablation 정의

## 원본

- 코드: `downstream/tasks/hypotension.py`, `downstream/run_ioh.py`
- BFM에서 surgical pull: `memory/project_physiome_state.md` "Surgical pull"
