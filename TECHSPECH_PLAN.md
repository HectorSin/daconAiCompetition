
# TECHSPEC_PLAN.md (v2 - Updated)

## 프로젝트 목표
데이터 공개 전, End-to-End 파이프라인(전처리, 특징 공학, 모델링, 검증)을 완성한다. 제공된 피드백을 반영하여, 더미 데이터로 전체 프로세스가 오류 없이 실행되는 견고한(Robust) 베이스라인을 구축한다.

---

## 1. 프로젝트 핵심 전략

### 과제 1: 공행성 (Comovement)
- 단순 CCF를 넘어 Granger Causality, DTW 등 고도화된 방법을 테스트한다.
- 9,900개 쌍에 대한 다중 검정 문제를 인지하고 제어한다 (FDR 적용).

### 과제 2: 예측 (Forecasting)
- 43개월의 짧은 데이터를 감안, 과적합 방지를 위해 LightGBM을 메인으로 한다.
- SARIMA/Prophet을 벤치마크로 사용한다.

### 견고성 (Robustness)
- `config.py`를 통한 설정 관리
- MLflow를 통한 실험 추적
- pytest를 통한 함수 단위 테스트로 재현성과 안정성을 확보한다.

---

## 2. 개발 환경 설정 (`requirements.txt`)
*Conda 가상환경 기준*

```
# Core
python==3.10.*
pandas
numpy
scikit-learn
jupyterlab

# Modeling (ML)
lightgbm
xgboost
catboost

# Modeling (Time-Series)
statsmodels  # CCF, ADF, KPSS, Granger, SARIMA
prophet      # Facebook Prophet
dtw-python   # Dynamic Time Warping
sktime       # STL Decomposition, Time-Series CV

# Hyperparameter Optimization
optuna

# Experiment Tracking
mlflow

# Testing
pytest

# Visualization
matplotlib
seaborn

# Utility
tqdm
pyarrow      # Parquet I/O
```

---

## 3. 프로젝트 디렉토리 구조 (v2)
*`config.py`와 `tests/` 디렉토리 추가*

```
/kookmin-trade-competition/
|
|-- /data/
|   |-- /raw/            # 원본 .csv
|   |-- /processed/      # 전처리된 .parquet
|
|-- /notebooks/
|   |-- 00_dummy_data_generator.ipynb  # (★1순위)
|   |-- 01_eda_and_preprocessing.ipynb # (ADF, KPSS, STL 테스트)
|   |-- 02_comovement_detection.ipynb  # (CCF, Granger, DTW 테스트)
|   |-- 03_forecasting_model.ipynb     # (LGBM, Prophet, SARIMA 학습)
|
|-- /src/
|   |-- preprocess.py      # 전처리, 데이터 품질 체크 함수
|   |-- features.py        # 특징 공학 함수 (Lag, Rolling, MoM, Fourier)
|   |-- comovement.py      # 공행성 탐지 함수 (CCF, Granger)
|   |-- model_wrappers.py  # 모델 래퍼
|   |-- train.py           # 전체 학습 파이프라인
|   |-- predict.py         # (필수) 추론 스크립트
|
|-- /tests/
|   |-- test_features.py   # 특징 공학 함수 단위 테스트
|   |-- test_preprocess.py # 전처리 함수 단위 테스트
|
|-- /models/               # 학습된 모델 (LGBM, Prophet)
|-- /output/               # 최종 제출 파일
|
|-- config.py              # (신규) 모든 설정 관리
|-- TECHSPEC_PLAN.md       # (현재 파일)
|-- mlflow.db              # (MLflow용)
|-- .gitignore
```

---

## 4. 설정 관리 (`config.py`)
*모든 경로, 파라미터, 시드를 중앙에서 관리*

```python
# config.py 예시
from pathlib import Path

class Config:
    # Seeds for reproducibility
    SEED = 42

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_RAW = BASE_DIR / "data" / "raw"
    DATA_PROCESSED = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "output"

    # Validation Strategy
    VAL_START_DATE = '2025-01-01'
    VAL_END_DATE = '2025-07-01'
    PRED_DATE = '2025-08-01'

    # Model Parameters
    LGBM_PARAMS = {
        'objective': 'regression_l1', # MAE
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'n_jobs': -1,
        'seed': SEED,
        'verbose': -1,
    }

    # Task 1: Comovement Detection Params
    CCF_LAG_MAX = 6  # 최대 6개월 선행까지 탐색
    CCF_THRESHOLD = 0.5 # CCF 임계값
    GRANGER_PVAL_THRESHOLD = 0.05 # Granger 인과관계 p-value 임계값
```

---

## 5. 가상 데이터 생성 (`00_dummy_data_generator.ipynb`)
*(기존 계획과 동일, 최우선 순위)*

- **스키마:** `date` (43개), `item_code` (100개), `value`
- **핵심:** 의도적인 공행성 쌍 주입 (예: `item_05(t) = item_01(t-2) * 0.7 + noise`)

---

## 6. 전처리 및 특징 공학 (`01`, `02`, `src/`)

### 6.A. 전처리 및 EDA 파이프라인 (Preprocessing & Quality Checks)
- **로드 및 피벗:** Long -> Wide, (43, 100)
- **데이터 품질 체크 (Risk Mgmt):**
    - `value`가 음수인 경우 확인 (무역량은 0 이상이어야 함).
    - 극단적 이상치 탐지 (IQR * 1.5) 및 로깅.
    - 결측 패턴 분석 (데이터 공개 후).
- **정상성 검증 (Stationarity Test):**
    - 100개 품목 모두에 대해 ADF Test, KPSS Test 실행 (statsmodels).
    - 결과를 저장하여 차분(differencing)이 필요한 품목 식별.
- **계절성 분해 (Seasonal Decomposition):**
    - STL Decomposition (sktime 또는 statsmodels) 실행.
    - `trend`, `seasonal`, `resid` 3개 요소를 분리하여 저장. (이후 피처로 활용)

### 6.B. 과제 1: 공행성 탐지 (Comovement Detection)
- **Baseline (CCF):** 시차 교차 상관계수 (statsmodels.ccf)를 Lag 1~6개월에 대해 계산.
- **Advanced 1 (Granger):** Granger Causality Test (statsmodels)로 인과 방향성 검증.
- **Advanced 2 (DTW):** Dynamic Time Warping (dtw-python)으로 비선형/가변 시차 관계 탐색.
- **다중 검정 문제 해결:**
    - 9,900개 쌍에 대한 p-value 리스트업.
    - Bonferroni Correction (너무 엄격할 수 있음) 또는 **FDR (Benjamini-Hochberg)**을 적용하여 False Positive 제어.

### 6.C. 과제 2: 예측 특징 공학 (Forecasting Feature Engineering)
*B 품목을 예측하기 위해 A (선행), B (후행) 품목 데이터를 활용.*

- **Lag Features:** A와 B의 1~12개월 전 `value`.
- **Date Features:** `month`, `year`, `quarter`.
- **Rolling Features:** A와 B의 3, 6, 12개월 이동 평균/표준편차.
- **Growth Rate:** MoM (`pct_change(1)`), YoY (`pct_change(12)`) 성장률.
- **Fourier Features:** `sin(2*pi*month/12)`, `cos(2*pi*month/12)` 등 계절성 패턴.
- **Decomposition Features:** STL 분해로 얻은 `trend`, `seasonal` 값.
- **Interaction Features:** `A_lag1 / B_lag1` (비율), `A_lag1 * B_lag1` (상호작용).

---

## 7. 모델링 및 검증 전략 (`03`, `src/`)

### 7.A. 검증 전략 (Validation)
- **Time-Series Cross-Validation (Sliding Window)** 사용 (sktime.forecasting.model_selection.SlidingWindowSplitter).
- **Local Valid Set:** `Config.VAL_START_DATE` ~ `Config.VAL_END_DATE` (2025.01 ~ 2025.07)
- **평가 지표:** RMSE, MAPE (추후 공개될 지표 대비)

### 7.B. 모델 후보군 (Model Candidates)
- **Simple Baseline:**
    - **SARIMA:** (statsmodels) 계절성을 고려한 전통적 시계열 모델.
    - **Prophet:** (prophet) FB에서 개발한 시계열 모델.
- **ML Baseline (Main):**
    - **LightGBM:** (Tabular 변환 후) 메인 모델.
- **Ensemble Candidates:**
    - XGBoost, CatBoost
