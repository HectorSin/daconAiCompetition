"""
초기 학습 파이프라인

더미 데이터를 로드하여 기본적인 학습 파이프라인을 구축합니다.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.preprocess import preprocess_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_features(df_wide: pd.DataFrame, target_item: str) -> pd.DataFrame:
    """
    간단한 특징을 생성합니다 (플레이스홀더).

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터프레임
    target_item : str
        예측할 타겟 품목

    Returns:
    --------
    pd.DataFrame
        특징 데이터프레임
    """
    features = pd.DataFrame(index=df_wide.index)

    # Lag features (1, 2, 3개월)
    for lag in [1, 2, 3]:
        features[f'{target_item}_lag_{lag}'] = df_wide[target_item].shift(lag)

    # Rolling mean (3개월)
    features[f'{target_item}_rolling_mean_3'] = df_wide[target_item].rolling(3).mean()

    # 월 정보
    features['month'] = df_wide.index.month

    # 결측값 제거
    features = features.dropna()

    return features


def train_dummy_model(data_path: Path):
    """
    더미 LightGBM 모델을 학습합니다.

    Parameters:
    -----------
    data_path : Path
        학습 데이터 경로
    """
    logger.info("=" * 50)
    logger.info("초기 학습 파이프라인 시작")
    logger.info("=" * 50)

    # 1. 데이터 로드 및 전처리
    logger.info("\n[1/5] 데이터 로드 중...")
    df_wide = preprocess_pipeline(data_path, check_quality=True)

    # 2. 특징 생성 (간단한 버전)
    logger.info("\n[2/5] 특징 생성 중...")
    target_item = 'item_00'  # 첫 번째 품목을 타겟으로 사용

    features = create_simple_features(df_wide, target_item)

    # 타겟 변수 생성
    y = df_wide[target_item].loc[features.index]

    logger.info(f"특징 shape: {features.shape}")
    logger.info(f"타겟 shape: {y.shape}")

    # 3. Train/Val Split
    logger.info("\n[3/5] Train/Val 분할 중...")
    # 시계열이므로 시간 순서 유지
    split_idx = int(len(features) * 0.8)

    X_train = features.iloc[:split_idx]
    X_val = features.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # 4. 모델 학습
    logger.info("\n[4/5] LightGBM 모델 학습 중...")

    model = LGBMRegressor(
        objective='regression',
        n_estimators=100,  # 더미용 적은 수
        learning_rate=0.05,
        num_leaves=31,
        random_state=Config.SEED,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse'
    )

    # 5. 평가
    logger.info("\n[5/5] 모델 평가 중...")

    # Train 예측
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    # Validation 예측
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)

    logger.info(f"\n{'=' * 50}")
    logger.info("평가 결과")
    logger.info(f"{'=' * 50}")
    logger.info(f"Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    logger.info(f"Val   RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
    logger.info(f"{'=' * 50}")

    # 6. 모델 저장
    model_path = Config.MODELS_DIR / "dummy_model.pkl"
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"\n✓ 모델 저장 완료: {model_path}")
    logger.info("\n✓ 초기 파이프라인 실행 완료!")

    return model


def main():
    """메인 함수"""
    data_path = Config.DATA_RAW / "dummy_trade_data.csv"

    if not data_path.exists():
        logger.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        logger.error("먼저 00_dummy_data_generator.ipynb를 실행하세요.")
        return

    train_dummy_model(data_path)


if __name__ == "__main__":
    main()
