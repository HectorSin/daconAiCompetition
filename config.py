"""
프로젝트 설정 관리 모듈

모든 경로, 파라미터, 시드를 중앙에서 관리합니다.
"""
from pathlib import Path


class Config:
    """프로젝트 전역 설정 클래스"""

    # 재현성을 위한 시드
    SEED = 42

    # 경로 설정
    BASE_DIR = Path(__file__).parent
    DATA_RAW = BASE_DIR / "data" / "raw"
    DATA_PROCESSED = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "output"
    NOTEBOOKS_DIR = BASE_DIR / "notebooks"

    # 검증 전략
    VAL_START_DATE = '2025-01-01'
    VAL_END_DATE = '2025-07-01'
    PRED_DATE = '2025-08-01'

    # 모델 파라미터
    LGBM_PARAMS = {
        'objective': 'regression_l1',  # MAE
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'n_jobs': -1,
        'seed': SEED,
        'verbose': -1,
    }

    # 과제 1: 공행성 탐지 파라미터
    CCF_LAG_MAX = 6  # 최대 6개월 선행까지 탐색
    CCF_THRESHOLD = 0.5  # CCF 임계값
    GRANGER_PVAL_THRESHOLD = 0.05  # Granger 인과관계 p-value 임계값
    GRANGER_MAX_LAG = 6  # Granger 검정 최대 시차
    DTW_THRESHOLD = 0.3  # DTW 거리 임계값 (정규화 후)

    # 다중 검정 보정
    FDR_ALPHA = 0.05  # FDR 유의수준

    # 특징 공학 파라미터
    LAG_FEATURES = [1, 2, 3, 6, 12]  # 생성할 lag 개수
    ROLLING_WINDOWS = [3, 6, 12]  # 이동 평균/표준편차 윈도우

    # MLflow 설정
    MLFLOW_TRACKING_URI = str(BASE_DIR / "mlruns")
    MLFLOW_EXPERIMENT_NAME = "trade-forecasting"
