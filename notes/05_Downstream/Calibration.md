---
tags: [downstream, calibration]
---

# Calibration — JBHI 가산점

## 왜 calibration인가

JBHI (clinical informatics venue)는 "AUROC만 높은 모델"보다 **operating point에서의 신뢰도**를 중요하게 본다. 임상 의사결정 (e.g., "60% 이상이면 alarm")이 calibration 없이는 의미 X.

## 측정 항목

| 항목 | 의미 | 계산 |
|---|---|---|
| **ECE** (Expected Calibration Error) | 예측 확률과 실제 빈도 간 평균 차이 | bin별 weighted mean of \|conf − acc\| |
| **Reliability diagram** | bin별 (예측 prob, 실제 positive rate) 산점도 | 대각선이 perfect calibration |
| **Sens@Sp90** | clinical operating point | specificity 90%에서 sensitivity |
| **Threshold report** | 운영 임계값 명시 | e.g., prob > 0.42 → alarm |

## 절차

```
1. Eval scripts에 --save_preds 옵션 활성화
2. results/<task>/seed_*/preds.npz 생성됨 (logit + true label per case)
3. downstream/calibration.py 실행
   → ECE, reliability plot, threshold curve 생성
4. (선택) Temperature scaling 적용 후 같은 항목 재측정
```

## Code

`downstream/calibration.py`:

```bash
python -m downstream.calibration \
    --preds results/ioh/seed_42/ABP+ECG+PPG/preds.npz \
    --output results/calibration/ioh_complete.png \
    --apply_temperature_scaling  # optional
```

## Boilerplate output

```
results/calibration/
├── ioh_{subset}_{seed}.png         — reliability diagram
├── ioh_{subset}_{seed}_ece.json    — ECE 수치
└── ioh_threshold_curve.png         — sens/spec at varying threshold
```

## 보고 권장

paper의 Discussion 섹션에:
- IOH의 7-subset 평균 ECE
- Complete vs single-modal에서의 ECE 변화 — **modality 결손 시 calibration이 무너지는가?** (clinical 신뢰도 직격)
- Temperature scaling이 효과적이면 final 모델에는 그것 적용된 prob 사용

## 실패 케이스 정성 분석 (선택)

JBHI 가산점의 또 다른 trick. saved preds + 원본 signal로:
- **High-confidence false positive** — IOH 발생 안 했는데 모델은 alarm 친 케이스
- **High-confidence false negative** — 실제 IOH 발생했는데 모델은 안전 판정

5–10 케이스의 signal plot + 임상적 해석 → "이 failure mode는 clinically interpretable" 한 사례 1–2개 찾으면 paper에 figure로 사용.

## 원본

- 코드: `downstream/calibration.py`
- 결정 메모: `memory/project_physiome_state.md` "Calibration / 실패 케이스 정성 분석"
