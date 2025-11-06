
# 상세 프로젝트 계획

이 계획은 `TECHSPEC_PLAN.md`를 상세하고 실행 가능한 체크리스트로 나눈 것입니다.

---

## 단계 1: 프로젝트 설정 및 기반 구축 (1-2일)

- [x] **1.1: 환경 설정**
    - [x] Conda 환경 생성
    - [x] `requirements.txt`의 모든 패키지 설치
    - [x] 주요 라이브러리(pandas, lightgbm, statsmodels, sktime)를 임포트하여 설치 확인

- [x] **1.2: 디렉토리 구조 생성**
    - [x] `data/raw`, `data/processed` 디렉토리 생성
    - [x] `notebooks` 디렉토리 생성
    - [x] `src` 디렉토리 생성
    - [x] `tests` 디렉토리 생성
    - [x] `models` 디렉토리 생성
    - [x] `output` 디렉토리 생성

- [x] **1.3: 설정 (`config.py`)**
    - [x] `config.py` 생성
    - [x] `Config` 클래스 구현
    - [x] `SEED` 추가
    - [x] 모든 디렉토리에 대한 `Path` 변수 추가
    - [x] 검증 및 예측 날짜 경계 추가
    - [x] `LGBM_PARAMS` 플레이스홀더 추가
    - [x] 공행성 탐지 파라미터 (`CCF_LAG_MAX` 등) 추가

---

## 단계 2: 더미 데이터 및 초기 파이프라인 (2-3일)

- [x] **2.1: 더미 데이터 생성 (`00_dummy_data_generator.ipynb`)**
    - [x] 노트북 생성
    - [x] `date` (43개월), `item_code` (100개), `value`를 포함하는 DataFrame 생성
    - [x] **중요:** 의도적인 공행성 쌍을 만드는 로직 구현 (예: `item_B(t) = f(item_A(t-k)) + noise`)
    - [x] 더미 데이터를 `data/raw/dummy_trade_data.csv`에 저장

- [x] **2.2: 초기 전처리 (`src/preprocess.py`)**
    - [x] `load_data` 함수 생성
    - [x] `pivot_data` 함수 생성 (long to wide 형식)
    - [x] **테스트:** `tests/test_preprocess.py`에 피벗 함수를 확인하는 간단한 테스트 작성

- [x] **2.3: 초기 학습 파이프라인 (`src/train.py`)**
    - [x] 기본 `main` 함수 생성
    - [x] `preprocess.py` 함수를 사용하여 더미 데이터 로드
    - [x] **플레이스홀더:** 더미 특징 공학 단계 생성
    - [x] **플레이스홀더:** 더미 `LightGBM` 모델 학습
    - [x] 스크립트가 오류 없이 처음부터 끝까지 실행되는지 확인

---

## 단계 3: 핵심 분석 - EDA 및 공행성 (3-4일)

- [ ] **3.1: EDA 및 전처리 (`01_eda_and_preprocessing.ipynb`, `src/preprocess.py`)**
    - [ ] **데이터 품질:**
        - [ ] `src/preprocess.py`에 `check_negative_values` 함수 구현
        - [ ] `src/preprocess.py`에 `log_outliers` 함수 (IQR 사용) 구현
    - [ ] **정상성:**
        - [ ] 100개 항목 모두에 대해 ADF 및 KPSS 테스트를 실행하는 함수를 `src`에 구현
        - [ ] 노트북에서 정상성 테스트 결과를 시각화하고 저장
        - [ ] 테스트 결과에 따라 비정상 항목에 차분 적용
    - [ ] **분해:**
        - [ ] STL 분해를 수행하는 함수를 `src`에 구현
        - [ ] 노트북에서 몇 가지 샘플 항목에 대한 추세, 계절성 및 잔차 구성 요소 시각화

- [ ] **3.2: 공행성 탐지 (`02_comovement_detection.ipynb`, `src/comovement.py`)**
    - [ ] **CCF:**
        - [ ] `src/comovement.py`에 `calculate_ccf` 함수 구현
        - [ ] 노트북에서 더미 데이터에 CCF를 실행하고 의도적으로 생성된 쌍 식별
    - [ ] **그레인저 인과관계:**
        - [ ] `src/comovement.py`에 `calculate_granger_causality` 함수 구현
        - [ ] 노트북에서 함수 테스트
    - [ ] **DTW:**
        - [ ] `src/comovement.py`에 `calculate_dtw` 함수 구현
        - [ ] 노트북에서 함수 테스트
    - [ ] **다중 검정 보정:**
        - [ ] p-value 목록에 FDR (Benjamini-Hochberg)을 적용하는 함수 구현
        - [ ] 9,900개 쌍에 걸쳐 CCF 및 그레인저 p-value 모두에 FDR 적용
        - [ ] 노트북의 공행성 탐지 워크플로우에 통합

---

## 단계 4: 특징 공학 및 모델링 (4-5일)

- [ ] **4.1: 특징 공학 (`src/features.py`)**
    - [ ] `generate_lag_features` 함수 생성
    - [ ] `generate_date_features` 함수 생성
    - [ ] `generate_rolling_features` 함수 생성
    - [ ] `generate_growth_rate_features` 함수 생성
    - [ ] `generate_fourier_features` 함수 생성
    - [ ] `generate_decomposition_features` 함수 생성
    - [ ] `generate_interaction_features` 함수 생성 (비율, 곱)
    - [ ] **테스트:** `tests/test_features.py`를 생성하고 최소 2-3개의 특징 생성 함수에 대한 단위 테스트 추가

- [ ] **4.2: 모델링 파이프라인 (`03_forecasting_model.ipynb`, `src/train.py`)**
    - [ ] `src/features.py`의 전체 특징 공학 파이프라인을 `src/train.py`에 통합
    - [ ] **검증:**
        - [ ] 학습 스크립트에 `sktime`의 `SlidingWindowSplitter` 구현
    - [ ] **모델:**
        - [ ] `src/model_wrappers.py`에 `LightGBM`용 래퍼 생성
        - [ ] 교차 검증 루프 내에서 `LightGBM` 모델 학습
        - [ ] 각 폴드에 대한 RMSE 및 MAPE 메트릭 기록
        - [ ] (선택 사항) 앙상블을 위해 XGBoost 및 CatBoost용 래퍼 생성
    - [ ] **벤치마킹:**
        - [ ] 노트북에서 단일 시계열에 대해 간단한 `SARIMA` 및 `Prophet` 모델을 학습하여 메인 모델과 비교

- [ ] **4.3: 예측 스크립트 (`src/predict.py`)**
    - [ ] `predict.py` 스크립트 생성
    - [ ] `/models` 디렉토리에서 학습된 모델을 로드해야 함
    - [ ] 새 데이터를 입력으로 받아 동일한 전처리 및 특징 공학을 적용하고 제출에 필요한 형식으로 출력 파일 생성

---

## 단계 5: 개선 및 최종화 (2-3일)

- [ ] **5.1: 하이퍼파라미터 튜닝**
    - [ ] `LGBM_PARAMS`를 튜닝하기 위해 `optuna`를 학습 스크립트 (`src/train.py`)에 통합
    - [ ] 더 나은 파라미터 세트를 찾기 위해 적은 수의 시도 실행
    - [ ] 찾은 최적의 파라미터로 `config.py` 업데이트

- [ ] **5.2: 실험 추적**
    - [ ] `mlflow` 로깅을 `src/train.py`에 통합
    - [ ] 각 실행에 대한 파라미터, 메트릭 및 학습된 모델 기록

- [ ] **5.3: 코드 검토 및 문서화**
    - [ ] `src/` 디렉토리의 모든 함수에 docstring 추가
    - [ ] 모든 코드가 명확성, 일관성 및 계획 준수 여부 검토
    - [ ] 설정 지침, 사용 가이드 및 파이프라인 개요로 `README.md` 생성/업데이트

- [ ] **5.4: 최종 실행**
    - [ ] 모든 더미 데이터 및 중간 파일 삭제
    - [ ] 최종적이고 깨끗한 코드로 처음부터 끝까지 전체 파이프라인 실행
