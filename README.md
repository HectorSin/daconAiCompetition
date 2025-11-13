# 데이콘 무역 예측 경진대회 프로젝트

제3회 국민대학교 AI빅데이터 분석 경진대회
국민은행 무역 데이터 분석 및 예측 프로젝트입니다.

## 프로젝트 구조

```
/daconai/
│
├── /data/
│   ├── /raw/            # 원본 데이터
│   └── /processed/      # 전처리된 데이터
│
├── /notebooks/          # Jupyter 노트북
│   ├── 00_dummy_data_generator.ipynb
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_comovement_detection.ipynb
│   └── 03_forecasting_model.ipynb
│
├── /src/                # 소스 코드
│   ├── preprocess.py
│   ├── features.py
│   ├── comovement.py
│   ├── model_wrappers.py
│   ├── train.py
│   └── predict.py
│
├── /tests/              # 단위 테스트
│   ├── test_preprocess.py
│   └── test_features.py
│
├── /models/             # 학습된 모델
├── /output/             # 최종 제출 파일
│
├── config.py            # 설정 관리
├── requirements.txt     # 패키지 의존성
├── PLAN.md             # 상세 프로젝트 계획
└── TECHSPEC_PLAN.md    # 기술 명세서
```

## 환경 설정

### 1. Conda 환경 생성 (Windows)

#### 방법 1: 자동 설치 스크립트 사용

```bash
setup_env.bat
```

#### 방법 2: 수동 설치

```bash
# 1. Conda 환경 생성
conda create -n daconai python=3.10 -y

# 2. 환경 활성화
conda activate daconai

# 3. 패키지 설치
pip install -r requirements.txt
```

### 2. 설치 검증

```bash
python verify_installation.py
```

모든 패키지가 정상적으로 설치되었는지 확인합니다.

### 3. 주요 라이브러리 테스트

```python
# Python 인터프리터에서 실행
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
import sktime

print("✓ 모든 라이브러리 import 성공!")
```

## 사용 방법 (PLAN.md 단계별 실행 가이드)

### 단계 1: 환경 설정 [완료 ✓]

```bash
# Conda 환경 생성
conda create -n daconai python=3.10 -y

# 환경 활성화
conda activate daconai

# 패키지 설치
pip install -r requirements.txt

# 설치 확인
python verify_installation.py
```

---

### 단계 2: 더미 데이터 및 초기 파이프라인 [완료 ✓]

```bash
# Jupyter Lab 실행 (더미 데이터 생성)
jupyter lab
# → notebooks/00_dummy_data_generator.ipynb 실행

# 초기 학습 파이프라인 실행
python src/train.py

# 단위 테스트 실행
/c/Users/SMART/anaconda3/envs/daconai/python.exe -m pytest tests/ -v

# 개별 테스트 파일
/c/Users/SMART/anaconda3/envs/daconai/python.exe -m pytest tests/test_preprocess.py -v
/c/Users/SMART/anaconda3/envs/daconai/python.exe -m pytest tests/test_comovement.py -v
/c/Users/SMART/anaconda3/envs/daconai/python.exe -m pytest tests/test_stationarity.py -v
```

---

### 단계 3: EDA 및 공행성 탐지 [완료 ✓] + 실제 데이터 EDA [진행 중]

```bash
# Jupyter Lab에서 EDA 노트북 실행
jupyter lab

# 실행할 노트북:
# → notebooks/01_eda_and_preprocessing.ipynb
#   - 실제 데이터 구조 확인
#   - 정상성 테스트 (ADF, KPSS)
#   - STL 분해 시각화
#   - 품목별 시계열 플롯

# → notebooks/02_comovement_detection.ipynb
#   - CCF 히트맵
#   - Granger 인과관계 네트워크
#   - DTW 클러스터링
#   - FDR 다중 검정 보정
```

---

### 단계 4: 특징 공학 및 모델링 [예정]

```bash
# 전체 모델링 파이프라인 실행
python src/train.py

# 학습된 모델 확인
ls models/

# 예측 수행
python src/predict.py
```

---

### 단계 5: 하이퍼파라미터 튜닝 및 MLflow [예정]

```bash
# MLflow UI 시작
mlflow ui
# → http://localhost:5000 접속

# Optuna 튜닝 실행 (구현 후)
python src/tune_hyperparams.py
```

---

### 빠른 참조: 주요 명령어

```bash
# 환경 활성화
conda activate daconai

# 테스트 실행
/c/Users/SMART/anaconda3/envs/daconai/python.exe -m pytest tests/ -v

# 학습 실행
python src/train.py

# Jupyter Lab
jupyter lab

# MLflow UI
mlflow ui
```

## 프로젝트 목표

1. **과제 1: 공행성 탐지**
   - CCF (Cross-Correlation Function)
   - Granger Causality Test
   - DTW (Dynamic Time Warping)
   - FDR (False Discovery Rate) 다중 검정 보정

2. **과제 2: 무역량 예측**
   - LightGBM 메인 모델
   - SARIMA, Prophet 벤치마크
   - Time-Series Cross-Validation

## 개발 진행 상황

- [x] **단계 1: 프로젝트 설정 및 기반 구축** (완료)
  - [x] Conda 환경 생성
  - [x] 패키지 설치 및 검증
  - [x] 프로젝트 구조 생성
  - [x] config.py 설정

- [x] **단계 2: 더미 데이터 및 초기 파이프라인** (완료)
  - [x] 더미 데이터 생성 (43개월, 100개 품목)
  - [x] 전처리 함수 구현
  - [x] 초기 학습 파이프라인
  - [x] 단위 테스트 작성 (24/25 통과)

- [x] **단계 3: 핵심 분석 - EDA 및 공행성** (완료)
  - [x] 정상성 테스트 (ADF, KPSS)
  - [x] STL 분해 구현
  - [x] CCF, Granger, DTW 공행성 탐지
  - [x] FDR 다중 검정 보정
  - [x] 실제 데이터 업로드 및 구조 분석

- [ ] **단계 4: 특징 공학 및 모델링** (진행 예정)
  - [ ] 실제 데이터 전처리 파이프라인 업데이트
  - [ ] Lag, Rolling, Growth Rate 특징 생성
  - [ ] LightGBM 모델 학습
  - [ ] Time-Series Cross-Validation
  - [ ] 예측 스크립트 작성

- [ ] **단계 5: 개선 및 최종화** (진행 예정)
  - [ ] Optuna 하이퍼파라미터 튜닝
  - [ ] MLflow 실험 추적
  - [ ] 최종 제출 파일 생성
  - [ ] 문서화 완료

자세한 계획은 [PLAN.md](PLAN.md)를 참조하세요.

### 현재 상태
- ✅ 단위 테스트: 24/25 통과 (96%)
- ✅ 실제 데이터: 업로드 완료 (10,836 rows, 100 items, 43 months)
- ⚠️ 데이터 이슈: 12.2% 결측값, 월별 다중 거래 (집계 필요)
- 📋 다음 단계: 실제 데이터 EDA 및 전처리 파이프라인 업데이트

자세한 테스트 결과는 [TEST_REPORT.md](TEST_REPORT.md)를 참조하세요.

## 실험 추적

MLflow를 사용하여 실험을 추적합니다:

```bash
# MLflow UI 실행
mlflow ui

# 브라우저에서 http://localhost:5000 접속
```

## 참고 문서

- [PLAN.md](PLAN.md) - 상세 실행 계획
- [TECHSPEC_PLAN.md](TECHSPEC_PLAN.md) - 기술 명세서
- [config.py](config.py) - 설정 관리

## 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.
