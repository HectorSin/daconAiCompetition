# 성능 향상 종합 계획

## 📊 현재 상태 분석

### 달성 사항
- ✅ 91/100개 품목 모델 학습 성공
- ✅ 평균 27.4개 특징 사용
- ✅ Train RMSE: 23,628 / MAE: 11,814
- ✅ 공행성 탐지: CCF, Granger, DTW 구현
- ✅ 제출 파일 생성 완료 (9,900행)
- ✅ **Forward Chaining CV 시스템 구축 (Phase 0 완료)**
- ✅ **Global Model 학습 완료 (Phase 1 핵심 완료)**
  - Val RMSE: 8,763.22 (±2,574.37)
  - Val MAE: 7,517.77 (±2,314.39)
  - 학습 샘플: 3,393개 (100개 품목 통합)

### 개선 필요 사항
- ❌ 9개 품목 학습 실패 (샘플 부족) → **Global Model로 해결 가능**
- ⚠️ 공행성 임계값 0.3 (29.1%만 활용) - 최적화 필요
- ⚠️ 하이퍼파라미터 튜닝 미실시 (다음 단계)
- ⚠️ 단일 모델만 사용 (앙상블 미적용) - Local + Global 앙상블 예정

---

## 🎯 성능 향상 전략 (우선순위별)

### 전략 1: 모델 성능 최적화 (최우선)

#### 1.1 하이퍼파라미터 튜닝 (Optuna)
- **목표:** LightGBM 파라미터 최적화로 RMSE/MAE 10-20% 개선
- **방법:**
  - Optuna를 사용한 베이지안 최적화
  - 탐색 파라미터:
    - `learning_rate`: [0.005, 0.1]
    - `num_leaves`: [15, 127]
    - `max_depth`: [3, 12]
    - `min_child_samples`: [5, 100]
    - `subsample`: [0.5, 1.0]
    - `colsample_bytree`: [0.5, 1.0]
    - `reg_alpha`: [0, 10]
    - `reg_lambda`: [0, 10]
  - 시계열 교차 검증과 결합
  - 품목별 최적 파라미터 저장
- **구현:**
  - `src/tune_hyperparameters.py` 생성
  - 상위 10개 품목으로 먼저 테스트
  - 최적 파라미터를 `config.py` 또는 `models/best_params.json`에 저장
- **예상 효과:** RMSE 15-20% 감소

#### 1.2 앙상블 모델링 (단순화 버전)
- **목표:** 다양한 모델 조합으로 예측 안정성 향상
- **방법:** **간단한 가중 평균부터 시작** (복잡한 스태킹은 나중에)
  - **Level 1 모델:**
    - Global LightGBM (메인)
    - Global XGBoost (보조)
    - 개별 LightGBM (보조)
  - **간단한 앙상블:**
    ```python
    def simple_ensemble(predictions_list, weights=None):
        if weights is None:
            weights = [1/len(predictions_list)] * len(predictions_list)
        return np.average(predictions_list, weights=weights, axis=0)
    ```
  - CV 성능 기반 가중치 자동 계산
- **구현:**
  - `src/simple_ensemble.py` 생성
- **예상 효과:** RMSE 3-7% 추가 감소 (스태킹보다 빠르고 안정적)

#### 1.3 시계열 교차 검증 강화
- **목표:** 과적합 방지 및 일반화 성능 향상
- **방법:**
  - Sliding Window CV (sktime) 적용
  - 3-5 Fold 시계열 분할
  - Out-of-Fold 예측으로 검증
- **구현:**
  - `src/cross_validation.py` 생성
  - `train_all_items.py`에 통합
- **예상 효과:** 일반화 성능 5-10% 향상

---

### 전략 2: 특징 공학 고도화

#### 2.1 공행성 임계값 최적화
- **목표:** 현재 0.3 임계값을 데이터 기반으로 최적화
- **방법:**
  - 임계값 0.1 ~ 0.7 범위에서 그리드 서치
  - 각 임계값별 예측 성능 평가
  - 최적 임계값 선택 (성능 vs 커버리지 트레이드오프)
- **구현:**
  - `notebooks/04_optimize_comovement_threshold.ipynb` 생성
  - `create_submission.py`에 최적 임계값 적용
- **예상 효과:** 제출 점수 5-15% 향상

#### 2.2 고급 특징 추가
- **목표:** 예측력 높은 새로운 특징 생성
- **추가 특징:**
  - **시계열 분해 특징:**
    - STL 분해의 `trend`, `seasonal`, `resid` 값
    - 추세 기울기, 계절성 강도
  - **통계 특징:**
    - 왜도(Skewness), 첨도(Kurtosis)
    - 변동계수(CV)
  - **공행성 강화 특징:**
    - 선행 품목의 2차, 3차 lag
    - 선행 품목과의 비율 변화율
    - DTW 거리 기반 유사도 점수
  - **외부 특징 (가능 시):**
    - 계절 더미 변수 강화
    - 경제 지표 (환율, GDP 등) - 데이터 확보 시
- **구현:**
  - `src/features.py`에 함수 추가
  - 특징 중요도 분석으로 선별
- **예상 효과:** 예측 정확도 10-15% 향상

#### 2.3 특징 선택 (Feature Selection)
- **목표:** 불필요한 특징 제거로 과적합 방지
- **방법:**
  - LightGBM Feature Importance 분석
  - Permutation Importance
  - SHAP Values
  - 상관관계 높은 특징 제거 (VIF \u003c 10)
- **구현:**
  - `notebooks/05_feature_selection.ipynb` 생성
  - 품목별 상위 20-30개 특징만 사용
- **예상 효과:** 과적합 감소, 학습 속도 향상

---

### 전략 3: 데이터 품질 개선

#### 3.1 샘플 부족 품목 처리 (Global Model) ⭐⭐⭐⭐⭐
- **목표:** 9개 실패 품목 학습 성공 + 전체 품목 성능 향상
- **방법:** **Global Model (Kaggle 우승 솔루션 정석)**
  - **Step 1:** 100개 품목 데이터를 하나의 DataFrame으로 통합
    - 컬럼 추가: `item_id` (Categorical Feature)
    - 형태: (43개월 × 100개 품목 = 4,300행)
  - **Step 2:** 하나의 거대 LightGBM 모델 학습
    - `item_id`를 categorical feature로 처리
    - 데이터 많은 품목의 패턴이 적은 품목에 전이
  - **Step 3:** 품목별 모델과 Global Model 앙상블
    - 샘플 충분한 품목: 개별 모델 70% + Global 30%
    - 샘플 부족한 품목: Global 100%
- **구현:**
  - `src/train_global_model.py` 생성
  - `src/ensemble_local_global.py` 생성
- **예상 효과:** 
  - 100개 품목 모두 예측 가능
  - 전체 성능 10-15% 향상 (데이터 부족 품목 특히 큰 개선)

#### 3.2 이상치 처리 강화
- **목표:** 극단값으로 인한 모델 왜곡 방지
- **방법:**
  - IQR 기반 이상치 탐지 (현재 구현됨)
  - Winsorization (상하위 1% 절삭)
  - Robust Scaler 적용
- **구현:**
  - `src/preprocess.py`에 함수 추가
  - 전처리 파이프라인에 통합
- **예상 효과:** 모델 안정성 향상

#### 3.3 결측치 처리 전략
- **목표:** 결측 패턴 분석 및 최적 대체
- **방법:**
  - 선형 보간 (Linear Interpolation)
  - 계절성 고려 보간 (Seasonal Decomposition)
  - KNN Imputer
  - 품목별 최적 방법 선택
- **구현:**
  - `src/imputation.py` 생성
  - 결측 패턴 분석 노트북 생성
- **예상 효과:** 데이터 활용률 향상

---

### 전략 4: 공행성 탐지 고도화

#### 4.1 다중 공행성 활용
- **목표:** 단일 선행 품목이 아닌 복수 선행 품목 활용
- **방법:**
  - 품목 B에 대해 상위 3-5개 선행 품목 식별
  - 가중 평균 또는 앙상블로 결합
  - 공행성 강도(CCF 값)를 가중치로 사용
- **구현:**
  - `src/comovement.py`에 함수 추가
  - `create_submission.py` 로직 수정
- **예상 효과:** 공행성 예측 정확도 15-25% 향상

#### 4.2 동적 임계값 적용
- **목표:** 품목별로 다른 임계값 사용
- **방법:**
  - 품목별 공행성 분포 분석
  - 상위 10%, 20%, 30% 등 백분위수 기반 임계값
  - 검증 성능 기반 최적 백분위수 선택
- **⚠️ Data Leakage 주의사항:**
  - **반드시 Train 기간 내 데이터로만 상관관계 계산**
  - Test 기간의 상관관계가 Train과 다를 수 있음
  - 너무 높은 상관관계(0.9+)만 믿는 것은 위험
  - 검증: Train/Val 기간별 상관관계 안정성 확인
- **구현:**
  - `notebooks/06_dynamic_threshold.ipynb` 생성
  - 품목별 임계값 딕셔너리 저장
  - Train 기간 상관관계 vs Val 기간 상관관계 비교
- **예상 효과:** 제출 점수 5-10% 향상

#### 4.3 인과관계 방향성 활용
- **목표:** Granger Causality 결과를 더 적극 활용
- **방법:**
  - CCF와 Granger 결과 결합
  - 양방향 인과관계 vs 단방향 구분
  - 인과 강도를 특징으로 추가
- **구현:**
  - `src/comovement.py` 개선
  - 인과관계 네트워크 시각화
- **예상 효과:** 공행성 탐지 정밀도 향상

---

### 전략 5: 실험 관리 및 재현성

#### 5.1 경량 실험 추적 시스템 (MLflow 대신)
- **목표:** 빠르고 간단한 실험 기록 및 비교
- **방법:** CSV 기반 로깅 (MLflow 서버 세팅 생략)
  - `experiments/log_experiment.csv` 파일 생성
  - 컬럼: `[timestamp, experiment_name, model_type, params, cv_score, test_score, notes]`
  - 각 실험마다 한 줄씩 append
  - pandas로 쉽게 읽고 비교 가능
- **구현:**
  - `src/utils/experiment_logger.py` 생성
    ```python
    def log_experiment(name, model_type, params, cv_score, test_score=None, notes=""):
        # CSV에 한 줄 추가
    ```
  - 모든 학습 스크립트에 통합
- **장점:**
  - 설정 시간 0분 (즉시 사용 가능)
  - 가볍고 빠름
  - Git으로 버전 관리 가능
- **예상 효과:** 실험 추적 효율성 2배 향상 (MLflow 대비 구축 시간 90% 절감)

#### 5.2 파이프라인 자동화
- **목표:** 전체 프로세스 원클릭 실행
- **방법:**
  - `main.py` 마스터 스크립트 생성
  - 전처리 → 특징 공학 → 학습 → 예측 → 제출 자동화
  - 설정 파일 기반 실행 (config.yaml)
- **구현:**
  - `main.py` 생성
  - `config.yaml` 생성
- **예상 효과:** 재현성 100% 보장

#### 5.3 코드 품질 및 문서화
- **목표:** 유지보수성 및 협업 효율성 향상
- **방법:**
  - 모든 함수에 docstring 추가
  - Type hints 추가
  - README.md 상세 작성
  - 주요 함수 단위 테스트 추가
- **구현:**
  - `tests/` 디렉토리 확장
  - `README.md` 업데이트
- **예상 효과:** 버그 감소, 협업 효율성 향상

---

## 📅 실행 계획 (우선순위별)

### Phase 0: 검증 전략 구축 (0.5-1일) - 🔴 최우선 🔴 ✅ **완료**
> **Critical:** 신뢰할 수 있는 검증 세트 없이는 모든 튜닝이 무의미합니다!

- [x] **1.3 시계열 교차 검증 강화** ⭐⭐⭐⭐⭐
  - [x] `src/cross_validation.py` 생성 (최우선!)
  - [x] **Forward Chaining CV 구현** (Sliding Window보다 안전)
    ```python
    # Forward Chaining 예시:
    # Train: [1:20] → Val: [21:23]
    # Train: [1:23] → Val: [24:26]
    # Train: [1:26] → Val: [27:29]
    # (미래 데이터 누출 위험 최소화)
    ```
  - [x] 3-5 Fold 시계열 분할 설정
  - [x] Out-of-Fold 예측 검증
  - [x] **중요:** Train 기간 내 데이터로만 검증 (Data Leakage 방지)
  - [x] CV 점수와 실제 성능 상관관계 확인
- [x] **경량 실험 로깅 시스템 구축**
  - [x] `src/utils/experiment_logger.py` 생성
  - [x] `experiments/log_experiment.csv` 초기화
  - [x] 간단한 로깅 함수: `log_experiment()`, `compare_experiments()`

### Phase 1: 즉시 실행 (1-2일) - 빠른 성과 ⭐

#### 🔴 최우선: Global Model 구축 ✅ **완료**
- [x] **3.1 Global Model 구축** (가장 큰 성능 향상!)
  - [x] `src/train_global_model.py` 생성
  - [x] 100개 품목 데이터를 하나의 DataFrame으로 통합
    - [x] `item_id`를 **categorical feature**로 지정
    - [x] 품목별 scale 차이 처리: 선택적 StandardScaler 적용
    - [x] 선행 품목 특징 추가 (CCF > 0.5, 상위 3개)
  - [x] Global LightGBM 학습 (CV 기반)
  - [x] Forward Chaining CV 구현 (n_splits=3, test_size=3)
  - [x] 모델 저장 및 실험 로깅
  
  **📊 학습 결과:**
  - 학습 샘플: 3,393개 (4,300개 → NaN 제거 후)
  - 특징 개수: 122개 (핵심 19개 + 선행 103개)
  - **평균 Val RMSE: 8,763.22 (±2,574.37)**
  - **평균 Val MAE: 7,517.77 (±2,314.39)**
  - 모델 저장: `models/global_model.pkl`
  
  **🔧 해결한 주요 문제:**
  1. **데이터 손실 문제 (100% 손실)**
     - 원인: Leading features가 99.1% NaN 생성 → `dropna()`로 모든 데이터 삭제
     - 해결: 선택적 NaN 처리
       - 핵심 특징(lag, rolling)만 NaN 검증
       - Leading features는 0으로 채움 (선행 관계 없음 표현)
  
  2. **CV 샘플 부족 문제**
     - 원인: 고정된 CV 파라미터 (n_splits=3, test_size=3)로 최소 19개 샘플 필요
     - 해결: 적응형 CV 파라미터
       - 30개 이상: 3×3
       - 20-29개: 2×2
       - 20개 미만: CV 건너뛰기
  
  3. **LightGBM categorical_feature 에러**
     - 원인: `categorical_feature`를 생성자에 전달하면 인식 안 됨
     - 해결: `fit()` 메서드에 직접 전달
       ```python
       fit_params = {'categorical_feature': ['item_id_encoded']}
       model.fit(X, y, **fit_params)
       ```
  
  **📁 생성된 파일:**
  - `src/train_global_model.py` - Global Model 학습 스크립트
  - `src/cross_validation.py` - Forward Chaining CV 시스템
  - `src/utils/experiment_logger.py` - 경량 실험 로깅
  - `models/global_model.pkl` - 학습된 Global Model
  - `experiments/log_experiment.csv` - 실험 기록

- [x] **앙상블 구현 완료 ✅**
  - [x] `src/ensemble_local_global.py` 생성
  - [x] 샘플 충분: Local 70% + Global 30%
  - [x] 샘플 부족: Global 100%
  - [x] 제출 및 점수 기록: **0.16054** (최고 성능)

#### 하이퍼파라미터 튜닝 (Global Model 대상)
- [ ] **1.1 Global Model 튜닝 (Optuna)**
  - [ ] `src/tune_global_model.py` 생성
  - [ ] Optuna trials 50-100으로 제한 (계산 자원 고려)
  - [ ] CV 기반 최적 파라미터 탐색
  - [ ] 최적 파라미터로 재학습

#### 공행성 안정성 검증 및 임계값 최적화
- [ ] **2.1 공행성 안정성 검증** (두 단계 접근)
  - [ ] `notebooks/04_comovement_stability.ipynb` 생성
  - [ ] **Step 1: 안정성 검증**
    - [ ] Train 전반부 vs 후반부 상관관계 비교
    - [ ] 안정적인 품목 쌍만 선별 (상관관계 변화 \< 0.2)
  - [ ] **Step 2: 임계값 최적화**
    - [ ] 안정적인 쌍에 대해서만 임계값 조정
    - [ ] 0.1 ~ 0.7 범위 그리드 서치
  - [ ] 최적 임계값으로 제출 파일 재생성

### Phase 2: 핵심 개선 (2-3일) - 성능 향상
- [ ] **1.2 앙상블 모델링**
  - [ ] `src/ensemble.py` 생성
  - [ ] LightGBM + XGBoost + CatBoost 앙상블
  - [ ] Weighted Average 적용
- [ ] **2.2 고급 특징 추가**
  - [ ] STL 분해 특징 추가
  - [ ] 통계 특징 추가
  - [ ] 공행성 강화 특징 추가
- [ ] **2.3 특징 선택**
  - [ ] `notebooks/05_feature_selection.ipynb` 생성
  - [ ] Feature Importance 분석
  - [ ] 상위 특징만 선택하여 재학습
- [ ] **4.1 다중 공행성 활용**
  - [ ] 복수 선행 품목 식별 로직 구현
  - [ ] 가중 평균 적용

### Phase 3: 고급 최적화 (2-3일) - 정밀 조정
- [ ] **4.2 동적 임계값 적용**
  - [ ] `notebooks/06_dynamic_threshold.ipynb` 생성
  - [ ] 품목별 최적 임계값 적용
- [ ] **4.3 인과관계 방향성 활용**
  - [ ] Granger + CCF 결합 로직 구현
  - [ ] 인과 네트워크 시각화

### Phase 4: 안정화 및 최종화 (1-2일) - 마무리
- [ ] **5.2 파이프라인 자동화**
  - [ ] `main.py` 생성
  - [ ] 전체 프로세스 자동화
- [ ] **5.3 코드 품질 및 문서화**
  - [ ] Docstring 추가
  - [ ] README.md 업데이트
  - [ ] 단위 테스트 추가
- [ ] **최종 실행**
  - [ ] 전체 파이프라인 처음부터 끝까지 실행
  - [ ] 최종 제출 파일 생성
  - [ ] 성능 검증 및 리포트 작성

---

## 🎯 예상 성능 향상

| 전략 | 예상 개선율 | 우선순위 | Phase |
|------|------------|---------|-------|
| **시계열 교차 검증** | **필수 기반** | ⭐⭐⭐⭐⭐ | **Phase 0** |
| 하이퍼파라미터 튜닝 | 15-20% | ⭐⭐⭐⭐⭐ | Phase 1 |
| 공행성 임계값 최적화 | 5-15% | ⭐⭐⭐⭐⭐ | Phase 1 |
| **Global Model** | **10-15%** | ⭐⭐⭐⭐⭐ | **Phase 1** |
| 앙상블 모델링 | 5-10% | ⭐⭐⭐⭐ | Phase 2 |
| 고급 특징 추가 | 10-15% | ⭐⭐⭐⭐ | Phase 2 |
| 다중 공행성 활용 | 15-25% | ⭐⭐⭐⭐ | Phase 2 |
| 특징 선택 | 5-10% | ⭐⭐⭐ | Phase 2 |
| 동적 임계값 | 5-10% | ⭐⭐⭐ | Phase 3 |

**총 예상 개선:** 35-60% (누적 효과)

> **🔴 Critical:** Phase 0 (시계열 교차 검증)은 모든 튜닝의 기반입니다. 이것 없이는 다른 모든 개선이 무의미할 수 있습니다!

---

## 🚀 실행 체크리스트 (간소화 버전)

### Day 1: 기반 구축 (6시간)
- [ ] **시계열 CV 구현** (2시간)
  - [ ] `src/cross_validation.py` - Forward Chaining CV
  - [ ] 더미 데이터로 검증
- [ ] **CSV 실험 로깅 구현** (30분)
  - [ ] `src/utils/experiment_logger.py`
  - [ ] 테스트 실행
- [ ] **Global Model 구축 시작** (3시간)
  - [ ] 100개 품목 데이터 통합
  - [ ] `item_id` categorical feature 설정
  - [ ] 첫 Global Model 학습

### Day 2-3: 핵심 개선 (필수)
- [ ] **Global Model 완성 및 검증**
  - [ ] 품목별 scale/계절성 처리
  - [ ] CV 기반 성능 검증
  - [ ] 9개 실패 품목 예측 확인
- [ ] **Optuna로 Global Model 튜닝**
  - [ ] 50-100 trials (계산 자원 고려)
  - [ ] 최적 파라미터 적용
- [ ] **공행성 안정성 검증 → 임계값 조정**
  - [ ] Train 전반부 vs 후반부 상관관계 비교
  - [ ] 안정적인 쌍만 선별
  - [ ] 임계값 최적화

### Day 4+: 추가 개선 (시간 여유시)
- [ ] **간단한 앙상블** (LGB + XGB)
  - [ ] 가중 평균 앙상블
  - [ ] CV 성능 기반 가중치
- [ ] **고급 특징 추가**
  - [ ] STL 분해 특징
  - [ ] 품목 간 상호작용 특징

### � 핵심 원칙
> **"Perfect is the enemy of good"**
> - Global Model 하나만 제대로 해도 상위권 가능
> - CV와 LB 점수 갭을 항상 모니터링 (과적합 방지)
> - 제출 횟수를 아끼지 마세요 - 실제 점수 피드백이 가장 중요

---

## 📝 참고 사항

### 🔴 Critical 주의사항
- **검증 전략 우선:** CV 없이 튜닝하면 과적합 위험 (Day 1 필수!)
- **Data Leakage 방지:** Train 기간 데이터로만 상관관계/임계값 계산
- **Global Model 우선:** 샘플 부족 품목은 Global Model이 가장 효과적

### 🚨 위험 요소 및 대응
1. **시간 부족 리스크**
   - 대응: Day 1 + Global Model만 해도 큰 개선 가능
   - Day 4+는 시간 여유시만 진행

2. **과적합 리스크**
   - 대응: **CV 점수가 너무 좋으면 오히려 의심**
   - Public LB 점수와 CV 점수 차이 모니터링
   - 차이가 크면 과적합 신호 → 정규화 강화

3. **계산 자원 부족**
   - 대응: Optuna trials 수를 50-100으로 제한
   - 품목별 튜닝 대신 Global Model 하나만 튜닝
## 🏆 제출 점수 추적 (Leaderboard Tracking)

> **목표:** 각 개선 사항의 실제 효과를 측정하고 최적의 전략을 찾습니다.

### 제출 기록

| # | 날짜 | 모델/전략 | CV Score | Public LB | Private LB | 비고 |
|---|------|---------|----------|-----------|------------|------|
| 1 | - | Baseline (91개 품목) | RMSE: 23,628 | 0.16037 | - | 초기 모델 |
| 2 | 2025-11-22 | Global Model | RMSE: 8,763 | 0.16022 | - | 과적합, 하락 ❌ |
| 3 | 2025-11-22 | Local 70% + Global 30% (CCF 0.3) | - | 0.16054 | - | 앙상블 효과 확인 |
| 4 | 2025-11-22 | **CCF 임계값 0.2** | - | **0.16807** | - | **대폭 상승 🚀** |

### 📊 결과 분석

**제출 #2: Global Model Baseline**
- CV RMSE: 8,763 (Baseline 23,628 대비 62.9% 개선)
- Public LB: 0.16022 (Baseline 0.16037 대비 0.09% 하락)
- **문제:** CV와 LB 간 큰 간극 → 과적합

### 📊 결과 분석

**제출 #4: 임계값 0.2 테스트 (앙상블) 🚀 대성공**
- Public LB: **0.16807** (기존 0.16054 대비 **4.7% 개선**)
- **발견:**
  - 임계값을 0.3 → 0.2로 낮추니 점수 급상승
  - 공행성 쌍: 3,233개 → 6,616개로 증가
  - **결론:** 놓치고 있던 공행성 관계가 많았음. 더 많은 쌍을 예측하는 것이 유리함.

**다음 개선 방향:**
1. **더 낮은 임계값 테스트** (긴급)
   - 0.15, 0.1, 0.05 테스트
   - 어디까지 낮췄을 때 성능이 오르는지 확인
   
2. **앙상블 가중치 최적화**
   - 현재 Local 70% 유지하면서 임계값 최적화 우선

### 다음 제출 계획

### 다음 제출 계획

#### 제출 #2: Global Model Baseline 🔴 **다음 단계**
- [ ] **Global Model로 예측 파일 생성**
  - [ ] `src/predict_global_model.py` 생성
  - [ ] Test 데이터 전처리
  - [ ] Global Model로 예측
  - [ ] 제출 파일 포맷 맞춰 저장
- [ ] **사이트에 제출**
- [ ] **점수 기록**
  - [ ] Public LB 점수 기록
  - [ ] CV 점수와 LB 점수 차이 분석
  - [ ] 과적합 여부 판단

#### 제출 #3: Local + Global 앙상블
- [ ] `src/ensemble_local_global.py` 구현
- [ ] 제출 및 점수 기록
- [ ] Baseline 대비 개선율 계산

#### 제출 #4: Optuna 튜닝 후
- [ ] Global Model 하이퍼파라미터 최적화
- [ ] 제출 및 점수 기록

#### 제출 #5: 공행성 임계값 최적화
- [ ] 임계값 0.1 ~ 0.7 범위 테스트
- [ ] 최적 임계값 적용
- [ ] 제출 및 점수 기록

### 성능 분석 가이드

**CV vs LB 점수 차이 해석:**
- **CV > LB:** 과적합 가능성 → 정규화 강화 필요
- **CV ≈ LB:** 좋은 일반화 → 현재 전략 유지
- **CV < LB:** 운이 좋거나 데이터 분포 차이 → CV 전략 재검토

**개선율 계산:**
```python
improvement = (baseline_score - new_score) / baseline_score * 100
print(f"개선율: {improvement:.2f}%")
```

---

## 📝 참고 사항
