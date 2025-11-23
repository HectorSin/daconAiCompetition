# 무역 품목 공행성 예측 - 추천 사전학습 모델 리스트

## 📊 대회 특성
- **도메인**: 무역 데이터 (100개 수입 품목)
- **Task**: 공행성 탐지 + 무역량 예측
- **데이터**: 2022.01 ~ 2025.07 (43개월)
- **특징**: 경제/수요 예측 유사

---

## 🎯 추천 사전학습 모델 (우선순위)

### 1. **Chronos** (Amazon) ⭐⭐⭐⭐⭐
- **장점**: 
  - 완전 오픈소스 & 무료
  - Zero-shot 성능 우수
  - 이미 구현 완료 (`src/train_chronos.py`)
- **설치**: 
  ```bash
  pip install git+https://github.com/amazon-science/chronos-forecasting.git
  ```
- **모델 크기**:
  - `chronos-t5-small` (8M) - 빠른 테스트용
  - `chronos-t5-base` (46M) - 균형
  - `chronos-t5-large` (200M) - 최고 성능 ✅ 추천
- **특징**: T5 아키텍처, 시계열을 토큰화하여 학습

---

### 2. **TimeGPT** (Nixtla) ⭐⭐⭐⭐
- **장점**:
  - **무역/수요 예측 특화** (retail, finance 데이터 학습)
  - 1000억+ 데이터 포인트 사전학습
  - API 방식으로 즉시 사용 가능
  - Multivariate (외부 변수 지원)
- **주의**: 
  - ⚠️ **API 기반 (유료)** - 무료 티어 500 requests/month
  - TimeGPT-2 출시 (2024) - 60% 정확도 향상
- **설치**:
  ```bash
  pip install nixtla
  ```
- **사용 예시**:
  ```python
  from nixtla import NixtlaClient
  client = NixtlaClient(api_key='YOUR_API_KEY')
  forecast = client.forecast(df, h=1)
  ```
- **무료 티어**: https://nixtla.io
- **적합 여부**: 
  - ✅ 무역 데이터에 강함 (경제 도메인)
  - ❌ 제출 20회 제약에서 API 비용 고려 필요

---

### 3. **TimesFM** (Google) ⭐⭐⭐⭐⭐
- **장점**:
  - 200M 파라미터
  - 1000억+ 시계열 데이터 사전학습
  - **무료 & 오픈소스**
  - 패치 기반 처리 (효율적)
- **설치**:
  ```bash
  pip install timesfm
  # 또는
  git clone https://github.com/google-research/timesfm
  ```
- **특징**: Decoder-only Transformer
- **적합 여부**: ✅ Chronos 대안/앙상블용

---

### 4. **Lag-Llama** (ServiceNow) ⭐⭐⭐⭐
- **장점**:
  - **Fine-tuning 쉬움** (무역 데이터에 적응)
  - 확률적 예측 (불확실성 제공)
  - 완전 오픈소스
- **설치**:
  ```bash
  pip install gluonts
  # Hugging Face에서 모델 로드
  ```
- **사용 예시**:
  ```python
  from gluonts.model.lag_llama import LagLlamaEstimator
  ```
- **적합 여부**: 
  - ✅ Fine-tuning으로 도메인 적응 가능
  - 📊 확률적 예측으로 리스크 관리

---

### 5. **Moirai** (Salesforce) ⭐⭐⭐
- **장점**: 다양한 주기 처리
- **오픈소스**: ✅
- **적합 여부**: 앙상블 추가용

---

### 6. **AutoGluon-TimeSeries** ⭐⭐⭐⭐
- **장점**:
  - **AutoML** - 자동 모델 선택 및 앙상블
  - Chronos, TimeGPT 등 통합
  - AWS 지원
- **설치**:
  ```bash
  pip install autogluon.timeseries
  ```
- **사용 예시**:
  ```python
  from autogluon.timeseries import TimeSeriesPredictor
  predictor = TimeSeriesPredictor().fit(train_data)
  predictions = predictor.predict(test_data)
  ```
- **적합 여부**: 
  - ✅ 빠른 베이스라인 구축
  - 📈 자동 앙상블

---

## 🎯 무역 데이터 특화 접근법

### 경제/금융 시계열 모델
1. **LSTM/GRU 기반 모델**
   - 경제 데이터의 장기 의존성 포착
   - 자체 학습 필요

2. **XGBoost/LightGBM** (전통적 방법)
   - 특징 공학과 결합 시 강력
   - 이미 구현 완료 (`src/train_global_model.py`)

3. **Hybrid 접근**
   - Foundation Model + 도메인 특징
   - **추천**: Chronos + 공행성 특징 + LightGBM

---

## 🚀 최종 추천 전략

### Scenario A: 빠른 구현 (2일)
```
Chronos (단독) 
→ 제출 #1
→ 점수 확인
```

### Scenario B: 앙상블 (3-4일) ⭐ 추천
```
Base Models:
├── Chronos (일반화)
├── N-HiTS (계층적 패턴)
└── LightGBM (공행성 특징)
    ↓
Stacking Meta-Learner
    ↓
최종 예측
```

### Scenario C: 최강 조합 (5일)
```
Base Models:
├── Chronos
├── TimesFM
├── Lag-Llama (Fine-tuned)
└── LightGBM
    ↓
Stacking Ensemble
    ↓
공행성 최적화
    ↓
최종 예측
```

---

## 📦 설치 명령어 모음

```bash
# 필수 (이미 실행)
pip install git+https://github.com/amazon-science/chronos-forecasting.git

# TimesFM (선택)
pip install timesfm

# Lag-Llama (선택)
pip install gluonts

# TimeGPT (API, 선택)
pip install nixtla

# AutoGluon (선택)
pip install autogluon.timeseries
```

---

## 🎬 다음 단계

### 단기 (오늘~내일)
1. ✅ Chronos 실행
   ```bash
   python src/train_chronos.py
   ```
2. ✅ 제출 #1
3. ✅ 앙상블 테스트
   ```bash
   python src/ensemble_chronos_nhits.py
   ```

### 중기 (2-3일차)
4. Stacking Ensemble
   ```bash
   python src/ensemble_stacking.py
   ```
5. TimesFM 추가 (시간 있으면)
6. Fine-tuning 실험

### 장기 (4-5일차)
7. 최적 조합 찾기
8. 공행성 임계값 최적화
9. 최종 제출

---

## 💡 주요 인사이트

### 무역 데이터 특성
- **경제 도메인** → TimeGPT, Chronos 유리
- **공행성 중요** → 전통적 특징 공학 필수
- **월별 데이터** → 계절성 강함 → N-HiTS 유용

### 앙상블 이유
- Foundation Model: 일반적 패턴 학습
- 도메인 모델: 무역 특화 특징 활용
- Meta-Learner: 자동 가중치 최적화

---

## 📚 참고 자료

- Chronos: https://github.com/amazon-science/chronos-forecasting
- TimeGPT: https://docs.nixtla.io
- TimesFM: https://github.com/google-research/timesfm
- Lag-Llama: https://huggingface.co/time-series-foundation-models/Lag-Llama
- AutoGluon: https://auto.gluon.ai/stable/tutorials/timeseries/

---

**추천 시작점**: Chronos → Stacking Ensemble → 제출!
