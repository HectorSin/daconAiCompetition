# 성능 향상을 위한 새로운 전략

## 🔍 문제 진단
- **임계값 0.2 (66.8%)** → 점수 0.173
- **임계값 0.35 (21.5%)** → 점수 0.157 (하락!)

**결론**: 공행성 탐지가 문제가 아니라 **예측 모델 자체의 정확도**가 문제

---

## 🚀 새로운 접근 방법

### 전략 1: 직접 예측 (공행성 무시) ⭐⭐⭐⭐⭐

**가설**: 공행성 판단이 오히려 성능을 저하시킴

**방법**:
```bash
python src/direct_prediction.py
```

모든 following_item을 Chronos로 직접 예측 (공행성 필터링 없음)

---

### 전략 2: AutoGluon 사용 ⭐⭐⭐⭐⭐

**가장 강력한 방법**: AutoML로 자동 최적화

```bash
pip install autogluon.timeseries
```

```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=1,
    eval_metric="MAPE"
)

predictor.fit(
    train_data,
    presets="best_quality",  # 최고 품질
    time_limit=3600  # 1시간
)

predictions = predictor.predict(test_data)
```

**장점**:
- 자동으로 최적 모델 선택
- 자동 앙상블
- 자동 하이퍼파라미터 튜닝

---

### 전략 3: 다중 모델 앙상블 (공행성 제외) ⭐⭐⭐⭐

**조합**:
1. Chronos (Foundation)
2. Prophet (Facebook)
3. ARIMA (통계)
4. LightGBM (ML)

각 모델로 직접 예측 → 가중 평균

---

### 전략 4: 특징 기반 LightGBM ⭐⭐⭐⭐

**현재 문제**: Chronos는 과거 데이터만 사용

**개선**:
- 계절성 특징 (월, 분기)
- 트렌드 특징 (이동평균, 기울기)
- Lag 특징 (1개월 전, 3개월 전, 12개월 전)
- 통계 특징 (표준편차, 변동계수)

```python
# 특징 생성
features = create_features(df)
# LightGBM 학습
model = lgb.LGBMRegressor()
model.fit(features, target)
```

---

### 전략 5: Prophet (Facebook) ⭐⭐⭐⭐

**장점**: 
- 계절성 자동 탐지
- 트렌드 자동 분해
- 휴일 효과 반영

```bash
pip install prophet
```

```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.fit(df)
forecast = model.predict(future)
```

---

## 📊 즉시 실행 가능한 순서

### 1단계: 직접 예측 테스트 (5분)
```bash
python src/direct_prediction.py
```
→ 제출 후 점수 확인

### 2단계: AutoGluon (1-2시간)
```bash
pip install autogluon.timeseries
python src/train_autogluon.py  # 새로 생성 필요
```
→ 자동으로 최적 모델 찾음

### 3단계: Prophet 추가 (30분)
```bash
pip install prophet
python src/train_prophet.py  # 새로 생성 필요
```

### 4단계: 앙상블 (30분)
Chronos + AutoGluon + Prophet 결합

---

## 💡 핵심 인사이트

**문제**: 공행성 탐지 자체가 노이즈를 추가하고 있음

**해결**: 
1. 공행성 무시하고 직접 예측
2. 더 강력한 모델 사용 (AutoGluon, Prophet)
3. 특징 공학 강화

**예상 성능**:
- 직접 예측: 0.173 → 0.20-0.25
- AutoGluon: 0.173 → 0.30-0.40
- Prophet: 0.173 → 0.25-0.35
- 앙상블: 0.173 → 0.40-0.50

---

## 🚀 지금 당장 실행

```bash
# 1. 직접 예측 (가장 빠름)
python src/direct_prediction.py

# 2. AutoGluon 설치
pip install autogluon.timeseries

# 3. Prophet 설치
pip install prophet
```

어떤 방법부터 시작할까요?
