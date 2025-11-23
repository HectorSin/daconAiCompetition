# Chronos Foundation Model 구현 계획

## 📊 현재 상황
- **현재 점수**: 0.17254 (N-HiTS)
- **목표 점수**: 0.43+ (본선 진출)
- **필요 개선**: 약 2.5배 성능 향상
- **제약**: 5일, 하루 4회 제출 = 총 20회

## 🎯 전략: Foundation Model + 앙상블

### 왜 Chronos + 앙상블인가?

#### ✅ Chronos의 장점
1. **Zero-shot 일반화**: 방대한 사전학습 데이터로 일반적 패턴 학습
2. **도메인 독립적**: 다양한 시계열 유형에 적용 가능
3. **확률적 예측**: 불확실성 고려 가능 (Monte Carlo)
4. **검증된 성능**: Amazon, Hugging Face 공식 모델

#### ✅ 앙상블의 장점
1. **상호 보완적**:
   - Chronos: 일반적 추세, 장기 패턴
   - N-HiTS: 계층적 보간, 복잡한 주기성
   - LightGBM: 공행성 특징 활용
2. **리스크 분산**: 단일 모델 실패 방지
3. **Kaggle 정석**: 대부분 우승 솔루션은 앙상블

---

## 📅 5일 실행 계획

### **Day 1: Chronos 기본 구현** (오늘)

#### 오전 (3시간)
- [ ] **환경 설정**
  ```bash
  pip install git+https://github.com/amazon-science/chronos-forecasting.git
  pip install torch transformers pandas
  ```
- [ ] **GPU 확인**
  ```python
  import torch
  print(torch.cuda.is_available())  # True여야 함
  ```

#### 오후 (4시간)
- [ ] **Chronos 실행**
  ```bash
  python src/train_chronos.py
  ```
  - 예상 시간: GPU 기준 30-60분 (100개 품목)
  - 출력: `submission_chronos_large.csv`

- [ ] **제출 #1: Chronos 단독**
  - 임계값 0.2 사용
  - 점수 기록 및 분석

#### 저녁 (2시간)
- [ ] **빠른 앙상블 테스트**
  ```bash
  python src/ensemble_chronos_nhits.py
  ```
  - 5개 가중치 조합 생성
  - **제출 #2-3**: 가장 유망한 2개 제출

---

### **Day 2: 앙상블 최적화**

#### 오전
- [ ] **Day 1 결과 분석**
  - Chronos vs N-HiTS vs 앙상블 비교
  - 최고 성능 가중치 식별

- [ ] **공행성 임계값 실험**
  - Chronos 기반으로 0.1, 0.15, 0.2, 0.25 테스트
  - **제출 #4-5**: 최적 임계값 탐색

#### 오후
- [ ] **Stacking Ensemble 구현**
  - `src/ensemble_stacking.py` 작성
  - Out-of-fold 예측 생성
  - Meta-learner 학습

- [ ] **제출 #6**: Stacking 앙상블

---

### **Day 3: Fine-tuning & 다양화**

#### 오전
- [ ] **Chronos 모델 크기 실험**
  - Base (46M) vs Large (200M) 비교
  - 빠른 small 모델도 테스트

- [ ] **제출 #7-8**: 다른 크기 모델

#### 오후
- [ ] **Fine-tuning (선택)**
  - 100개 품목 데이터로 adaptation
  - Low-rank adapter 사용
  - **주의**: 과적합 위험, CV로 검증 필수

- [ ] **제출 #9**: Fine-tuned 모델 (검증 후)

---

### **Day 4: 고급 앙상블 & 특징**

#### 오전
- [ ] **TimesFM 추가** (시간 있으면)
  - Google TimesFM 모델 테스트
  - 3-way 앙상블: Chronos + N-HiTS + TimesFM

- [ ] **제출 #10-11**: 3-way 앙상블

#### 오후
- [ ] **공행성 특징 강화**
  - 다중 선행 품목 활용
  - DTW 거리 기반 가중치

- [ ] **제출 #12-13**: 강화된 공행성

---

### **Day 5: 최종 최적화**

#### 오전
- [ ] **지금까지 최고 조합 찾기**
  - 모든 제출 점수 분석
  - Top 3 모델 식별

- [ ] **앙상블 가중치 미세 조정**
  - Top 3를 다양한 비율로 결합
  - **제출 #14-16**

#### 오후
- [ ] **최종 제출 전략**
  - Public LB 점수 기반 선택
  - Private LB 대비 안전한 조합
  - **제출 #17-20**: 최종 후보들

---

## 🔧 구현된 스크립트

### 1. `src/train_chronos.py`
- **기능**: Chronos-T5-large로 100개 품목 예측
- **입력**: `data/raw/train.csv`
- **출력**: `output/submission_log/{timestamp}/submission_chronos_large.csv`
- **시간**: GPU 기준 30-60분

**주요 특징**:
- Zero-shot 예측 (학습 없음)
- 확률적 예측 (20 samples → median)
- CCF 공행성 결합

### 2. `src/ensemble_chronos_nhits.py`
- **기능**: Chronos + N-HiTS 가중 평균 앙상블
- **입력**: 두 모델의 제출 파일
- **출력**: 5개 가중치 조합 파일
- **시간**: 1분 미만

**가중치 조합**:
- `5c_5n`: 50% / 50% (균형)
- `6c_4n`: 60% / 40% (Chronos 우세) ⭐ 추천
- `7c_3n`: 70% / 30% (Chronos 강우세)
- `4c_6n`: 40% / 60% (N-HiTS 우세)
- `8c_2n`: 80% / 20% (Chronos 매우 우세)

---

## 📈 예상 성능 시나리오

### 시나리오 A: 보수적 (60% 확률)
```
Day 1: Chronos → 0.20 (16% 향상)
Day 2: Ensemble → 0.25 (45% 향상)
Day 3: Fine-tune → 0.28 (62% 향상)
Day 4-5: 최적화 → 0.30-0.35
```
**결과**: 본선 진출 실패 😞

### 시나리오 B: 기대 (30% 확률)
```
Day 1: Chronos → 0.25 (45% 향상)
Day 2: Ensemble → 0.35 (103% 향상)
Day 3: Fine-tune → 0.40 (132% 향상)
Day 4-5: 최적화 → 0.43-0.45
```
**결과**: 본선 진출 성공! 🎉

### 시나리오 C: 최상 (10% 확률)
```
Day 1: Chronos → 0.30 (74% 향상)
Day 2: Ensemble → 0.42 (144% 향상)
Day 3: Fine-tune → 0.48 (178% 향상)
Day 4-5: 최적화 → 0.50+
```
**결과**: 상위권 진입! 🚀

---

## 💡 핵심 성공 요인

### 1. **빠른 실행** ⚡
- Day 1 오늘 안에 Chronos 제출!
- 매일 최소 2-3회 제출

### 2. **체계적 실험** 📊
- 모든 제출을 `experiments/log_experiment.csv`에 기록
- 점수 변화 패턴 분석

### 3. **리스크 관리** 🛡️
- Fine-tuning은 CV 검증 후에만
- Public LB 과적합 주의
- 안전한 조합도 제출

### 4. **앙상블 우선** 🎯
- 단일 모델보다 앙상블에 집중
- 다양한 가중치 실험

---

## 🚨 주의사항

### Data Leakage 방지
- CCF 계산은 Train 기간만 사용
- Test 데이터 절대 참조 금지

### GPU 메모리 관리
- Chronos-large는 ~8GB VRAM 필요
- 메모리 부족 시 base 또는 small 사용

### 제출 횟수 관리
| Day | 제출 계획 | 누적 | 여유 |
|-----|----------|------|------|
| 1   | 3회      | 3    | 17   |
| 2   | 3회      | 6    | 14   |
| 3   | 3회      | 9    | 11   |
| 4   | 4회      | 13   | 7    |
| 5   | 7회      | 20   | 0    |

---

## 🎬 지금 바로 시작하기

### Step 1: Chronos 설치
```bash
conda activate your_env
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

### Step 2: GPU 확인
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Step 3: 첫 예측 실행
```bash
cd c:\Users\Jamtol\git\daconAI
python src/train_chronos.py
```

### Step 4: 제출!
- 생성된 CSV 파일을 대회 사이트에 제출
- 점수 기록

---

## 📚 참고 자료

### Chronos 공식 문서
- GitHub: https://github.com/amazon-science/chronos-forecasting
- Paper: https://arxiv.org/abs/2403.07815
- Hugging Face: https://huggingface.co/amazon/chronos-t5-large

### 앙상블 Best Practices
- Stacking: Meta-learner로 가중치 자동 학습
- Weighted Average: 빠르고 효과적
- Blending: Train/Val 분리 후 결합

---

## ✅ 다음 단계

바로 실행:
```bash
python src/train_chronos.py
```

예상 소요 시간: 30-60분 (GPU 기준)

완료 후:
1. 제출 파일 확인: `output/submission_log/{timestamp}/`
2. 대회 사이트 제출
3. 점수 기록
4. 앙상블 실행: `python src/ensemble_chronos_nhits.py`

**화이팅! 🚀**
