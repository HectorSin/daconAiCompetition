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

## 사용 방법

### 1. 더미 데이터 생성

```bash
# Jupyter Lab 실행
jupyter lab

# notebooks/00_dummy_data_generator.ipynb 실행
```

### 2. 초기 파이프라인 테스트

```bash
python src/train.py
```

### 3. 테스트 실행

```bash
# 전체 테스트 실행
pytest tests/

# 특정 테스트 파일 실행
pytest tests/test_preprocess.py -v
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

- [x] 단계 1: 프로젝트 설정 및 기반 구축
- [x] 단계 2: 더미 데이터 및 초기 파이프라인
- [ ] 단계 3: 핵심 분석 - EDA 및 공행성
- [ ] 단계 4: 특징 공학 및 모델링
- [ ] 단계 5: 개선 및 최종화

자세한 계획은 [PLAN.md](PLAN.md)를 참조하세요.

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
