"""
전체 품목에 대한 모델 학습 스크립트

각 품목별로 독립적인 예측 모델을 학습합니다.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.features import create_features_for_item, prepare_train_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model_for_item(df_wide: pd.DataFrame,
                         item: str,
                         ccf_results: pd.DataFrame,
                         train_size: float = 0.8) -> dict:
    """
    단일 품목에 대한 모델을 학습합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터
    item : str
        품목명
    ccf_results : pd.DataFrame
        CCF 결과
    train_size : float
        학습 데이터 비율

    Returns:
    --------
    dict
        학습된 모델과 메트릭 정보
    """
    # 1. 특징 생성
    features = create_features_for_item(
        df_wide, item, ccf_results,
        lag_features=[1, 2, 3, 6],
        rolling_windows=[3, 6],
        include_leading=True,
        top_leading=3
    )

    # 2. 학습 데이터 준비
    X, y = prepare_train_data(features, item, min_samples=12)

    if len(X) < 12:
        logger.warning(f"품목 {item}: 샘플이 부족하여 학습 불가 ({len(X)}개)")
        return None

    # 3. Train/Val Split (시계열이므로 시간 순서 유지)
    split_idx = int(len(X) * train_size)

    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]

    # 4. 모델 학습
    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=Config.SEED,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)] if len(X_val) > 0 else None,
        callbacks=[],
        eval_metric='rmse'
    )

    # 5. 평가
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    if len(X_val) > 0:
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
    else:
        val_rmse = None
        val_mae = None

    return {
        'model': model,
        'feature_names': X.columns.tolist(),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'val_rmse': val_rmse,
        'val_mae': val_mae
    }


def main():
    print("=" * 60)
    print("전체 품목 모델 학습")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/4] 데이터 로드 중...")
    df_raw = pd.read_csv(Config.DATA_RAW / 'train.csv')
    df_raw['date'] = pd.to_datetime(df_raw[['year', 'month']].assign(day=1))

    df_agg = df_raw.groupby(['date', 'item_id']).agg({
        'value': 'sum'
    }).reset_index()

    df_wide = df_agg.pivot(index='date', columns='item_id', values='value').fillna(0)
    print(f"데이터 shape: {df_wide.shape}")

    # 2. CCF 결과 로드
    print("\n[2/4] CCF 결과 로드 중...")
    ccf_results = pd.read_csv(Config.DATA_PROCESSED / 'ccf_results.csv')
    print(f"CCF 결과: {len(ccf_results)}개 쌍")

    # 3. 전체 품목 학습
    print(f"\n[3/4] 전체 {len(df_wide.columns)}개 품목 학습 중...")

    models = {}
    metrics = []

    for idx, item in enumerate(df_wide.columns):
        if (idx + 1) % 10 == 0:
            print(f"진행: {idx+1}/{len(df_wide.columns)} 품목...")

        try:
            result = train_model_for_item(df_wide, item, ccf_results, train_size=0.8)

            if result is not None:
                models[item] = result
                metrics.append({
                    'item': item,
                    'train_samples': result['train_samples'],
                    'val_samples': result['val_samples'],
                    'train_rmse': result['train_rmse'],
                    'train_mae': result['train_mae'],
                    'val_rmse': result['val_rmse'],
                    'val_mae': result['val_mae'],
                    'n_features': len(result['feature_names'])
                })
        except Exception as e:
            logger.error(f"품목 {item} 학습 실패: {e}")
            continue

    print(f"\n학습 완료: {len(models)}개 품목")

    # 4. 모델 저장
    print("\n[4/4] 모델 저장 중...")
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 개별 모델 저장
    for item, result in models.items():
        model_path = Config.MODELS_DIR / f'model_{item}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)

    # 메트릭 저장
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(Config.MODELS_DIR / 'training_metrics.csv', index=False)

    print(f"모델 저장 완료: {Config.MODELS_DIR}")

    # 요약 통계
    print("\n" + "=" * 60)
    print("학습 요약")
    print("=" * 60)
    print(f"총 품목: {len(df_wide.columns)}개")
    print(f"학습 성공: {len(models)}개")
    print(f"학습 실패: {len(df_wide.columns) - len(models)}개")

    if len(metrics) > 0:
        print(f"\n평균 메트릭:")
        print(f"  Train RMSE: {metrics_df['train_rmse'].mean():.2f}")
        print(f"  Train MAE: {metrics_df['train_mae'].mean():.2f}")
        print(f"  Val RMSE: {metrics_df['val_rmse'].mean():.2f} (유효한 경우)")
        print(f"  Val MAE: {metrics_df['val_mae'].mean():.2f} (유효한 경우)")
        print(f"  평균 특징 개수: {metrics_df['n_features'].mean():.1f}개")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
