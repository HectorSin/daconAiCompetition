"""
Local + Global 앙상블 제출 파일 생성

샘플 충분한 품목: Local 70% + Global 30%
샘플 부족한 품목: Global 100%
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.features import create_features_for_item

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_with_local_model(df_wide: pd.DataFrame,
                             item: str,
                             ccf_results: pd.DataFrame,
                             model_path: Path) -> float:
    """
    Local 모델로 품목의 2025.08 예측값 계산
    """
    try:
        with open(model_path, 'rb') as f:
            result = pickle.load(f)
        model = result['model']
        feature_names = result['feature_names']
    except Exception as e:
        logger.warning(f"품목 {item} Local 모델 로드 실패: {e}")
        return None

    # 특징 생성
    features = create_features_for_item(
        df_wide, item, ccf_results,
        lag_features=[1, 2, 3, 6],
        rolling_windows=[3, 6],
        include_leading=True,
        top_leading=3
    )

    # 마지막 행 (2025.07의 특징 = 2025.08 예측용)
    last_row = features.iloc[[-1]]

    # 필요한 특징만 선택
    try:
        X_pred = last_row[feature_names]
    except KeyError:
        logger.warning(f"품목 {item}: 특징 불일치")
        return None

    # 결측값 체크
    if X_pred.isnull().any().any():
        logger.warning(f"품목 {item}: 예측 특징에 결측값 존재")
        return None

    # 예측
    pred = model.predict(X_pred)[0]
    pred = max(0, pred)

    return pred


def predict_with_global_model_single_item(df_wide: pd.DataFrame,
                                          item: str,
                                          ccf_results: pd.DataFrame,
                                          model_data: dict) -> float:
    """
    Global 모델로 단일 품목의 2025.08 예측값 계산
    """
    model = model_data['model']
    feature_names = model_data['feature_names']
    label_encoder = model_data['label_encoder']
    
    # 학습되지 않은 품목 체크
    if item not in label_encoder.classes_:
        return None
    
    target_date = pd.Timestamp('2025-08-01')
    row_data = {'item_id': item}
    
    # 1. Lag 특징
    for lag in [1, 2, 3, 6]:
        lag_date = target_date - pd.DateOffset(months=lag)
        if lag_date in df_wide.index and item in df_wide.columns:
            row_data[f'lag_{lag}'] = df_wide.loc[lag_date, item]
        else:
            row_data[f'lag_{lag}'] = np.nan
    
    # 2. Rolling 특징
    for window in [3, 6]:
        start_date = target_date - pd.DateOffset(months=window)
        if item in df_wide.columns:
            recent_data = df_wide.loc[
                (df_wide.index > start_date) & (df_wide.index < target_date),
                item
            ]
            if len(recent_data) > 0:
                row_data[f'rolling_mean_{window}'] = recent_data.mean()
                row_data[f'rolling_std_{window}'] = recent_data.std()
                row_data[f'rolling_min_{window}'] = recent_data.min()
                row_data[f'rolling_max_{window}'] = recent_data.max()
            else:
                row_data[f'rolling_mean_{window}'] = np.nan
                row_data[f'rolling_std_{window}'] = np.nan
                row_data[f'rolling_min_{window}'] = np.nan
                row_data[f'rolling_max_{window}'] = np.nan
    
    # 3. 성장률 특징
    if 'lag_1' in row_data and not pd.isna(row_data['lag_1']):
        if 'lag_2' in row_data and not pd.isna(row_data['lag_2']):
            row_data['diff_1'] = row_data['lag_1'] - row_data['lag_2']
            if row_data['lag_2'] != 0:
                row_data['pct_change_1'] = (row_data['lag_1'] - row_data['lag_2']) / row_data['lag_2'] * 100
            else:
                row_data['pct_change_1'] = 0
        else:
            row_data['diff_1'] = np.nan
            row_data['pct_change_1'] = np.nan
    else:
        row_data['diff_1'] = np.nan
        row_data['pct_change_1'] = np.nan
    
    # 4. 날짜 특징
    row_data['month'] = 8
    row_data['quarter'] = 3
    row_data['year'] = 2025
    row_data['month_sin'] = np.sin(2 * np.pi * 8 / 12)
    row_data['month_cos'] = np.cos(2 * np.pi * 8 / 12)
    
    # 5. 선행 품목 특징
    leading_pairs = ccf_results[
        (ccf_results['item_y'] == item) &
        (ccf_results['abs_ccf'] >= 0.5)
    ].nlargest(3, 'abs_ccf')
    
    for idx, pair_row in leading_pairs.iterrows():
        leading_item = pair_row['item_x']
        optimal_lag = int(pair_row['optimal_lag'])
        
        if leading_item in df_wide.columns:
            lag_date = target_date - pd.DateOffset(months=optimal_lag)
            col_name = f'leading_{leading_item}_lag_{optimal_lag}'
            
            if lag_date in df_wide.index:
                row_data[col_name] = df_wide.loc[lag_date, leading_item]
            else:
                row_data[col_name] = np.nan
    
    # DataFrame으로 변환
    row_df = pd.DataFrame([row_data])
    
    # 핵심 특징만 NaN 체크
    core_cols = [col for col in row_df.columns 
                if not col.startswith('leading_') and col != 'item_id']
    
    if row_df[core_cols].isnull().any().any():
        return None
    
    # 선행 특징 NaN은 0으로 채움
    leading_cols = [col for col in row_df.columns if col.startswith('leading_')]
    if leading_cols:
        row_df[leading_cols] = row_df[leading_cols].fillna(0)
    
    # item_id 인코딩
    row_df['item_id_encoded'] = label_encoder.transform([item])[0]
    
    # 누락된 특징은 0으로 채움
    for col in feature_names:
        if col not in row_df.columns:
            row_df[col] = 0
    
    # 예측
    X = row_df[feature_names]
    pred = model.predict(X)[0]
    pred = max(0, pred)
    
    return pred


def create_ensemble_predictions(df_wide: pd.DataFrame,
                                ccf_results: pd.DataFrame,
                                local_models_dir: Path,
                                global_model_path: Path,
                                local_weight: float = 0.7) -> dict:
    """
    Local + Global 앙상블 예측값 생성
    
    Parameters:
    -----------
    local_weight : float
        Local 모델 가중치 (Global은 1 - local_weight)
    
    Returns:
    --------
    dict
        {item_id: predicted_value}
    """
    logger.info("Local + Global 앙상블 예측 시작...")
    logger.info(f"가중치: Local {local_weight*100:.0f}% + Global {(1-local_weight)*100:.0f}%")
    
    # Global 모델 로드
    with open(global_model_path, 'rb') as f:
        global_model_data = pickle.load(f)
    
    predictions = {}
    local_count = 0
    global_only_count = 0
    ensemble_count = 0
    
    for item in df_wide.columns:
        local_model_path = local_models_dir / f'model_{item}.pkl'
        
        # Local 모델 예측
        local_pred = None
        if local_model_path.exists():
            local_pred = predict_with_local_model(df_wide, item, ccf_results, local_model_path)
        
        # Global 모델 예측
        global_pred = predict_with_global_model_single_item(df_wide, item, ccf_results, global_model_data)
        
        # 앙상블
        if local_pred is not None and global_pred is not None:
            # 둘 다 있으면 가중 평균
            final_pred = local_weight * local_pred + (1 - local_weight) * global_pred
            ensemble_count += 1
        elif local_pred is not None:
            # Local만 있으면 Local 사용
            final_pred = local_pred
            local_count += 1
        elif global_pred is not None:
            # Global만 있으면 Global 사용
            final_pred = global_pred
            global_only_count += 1
        else:
            # 둘 다 없으면 0
            final_pred = 0.0
        
        predictions[item] = final_pred
    
    logger.info(f"\n예측 완료:")
    logger.info(f"  - 앙상블 (Local + Global): {ensemble_count}개")
    logger.info(f"  - Local만: {local_count}개")
    logger.info(f"  - Global만: {global_only_count}개")
    logger.info(f"  - 총 품목: {len(predictions)}개")
    logger.info(f"  - 예측값 범위: {min(predictions.values()):.2f} ~ {max(predictions.values()):.2f}")
    logger.info(f"  - 예측값 평균: {np.mean(list(predictions.values())):.2f}")
    
    return predictions


def create_submission_from_ensemble(predictions: dict,
                                    ccf_results: pd.DataFrame,
                                    sample_submission_path: Path,
                                    output_path: Path,
                                    ccf_threshold: float = 0.3) -> pd.DataFrame:
    """
    앙상블 예측값으로 제출 파일 생성
    """
    logger.info("제출 파일 생성 중...")
    
    submission = pd.read_csv(sample_submission_path)
    logger.info(f"샘플 제출 파일: {len(submission)}개 쌍")
    
    values = []
    
    for idx, row in submission.iterrows():
        leading_item = row['leading_item_id']
        following_item = row['following_item_id']
        
        # CCF 결과에서 공행성 확인
        comovement = ccf_results[
            ((ccf_results['item_x'] == leading_item) & (ccf_results['item_y'] == following_item)) |
            ((ccf_results['item_x'] == following_item) & (ccf_results['item_y'] == leading_item))
        ]
        
        has_comovement = False
        if len(comovement) > 0:
            max_ccf = comovement['abs_ccf'].max()
            if max_ccf >= ccf_threshold:
                has_comovement = True
        
        if has_comovement:
            value = predictions.get(following_item, 0.0)
        else:
            value = 0.0
        
        values.append(value)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"진행: {idx+1}/{len(submission)}")
    
    submission['value'] = values
    submission.to_csv(output_path, index=False)
    
    logger.info(f"제출 파일 저장 완료: {output_path}")
    
    # 통계
    n_nonzero = (submission['value'] > 0).sum()
    n_zero = (submission['value'] == 0).sum()
    
    logger.info(f"\n제출 파일 통계:")
    logger.info(f"  총 쌍: {len(submission)}개")
    logger.info(f"  공행성 있음 (value > 0): {n_nonzero}개 ({n_nonzero/len(submission)*100:.1f}%)")
    logger.info(f"  공행성 없음 (value = 0): {n_zero}개 ({n_zero/len(submission)*100:.1f}%)")
    logger.info(f"  value 합계: {submission['value'].sum():,.0f}")
    logger.info(f"  value 평균: {submission['value'].mean():,.2f}")
    
    return submission


def main():
    print("=" * 60)
    print("Local + Global 앙상블 제출 파일 생성")
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
    
    # 3. 앙상블 예측
    print("\n[3/4] Local + Global 앙상블 예측 중...")
    predictions = create_ensemble_predictions(
        df_wide=df_wide,
        ccf_results=ccf_results,
        local_models_dir=Config.MODELS_DIR,
        global_model_path=Config.MODELS_DIR / 'global_model.pkl',
        local_weight=0.7  # Local 70% + Global 30%
    )
    
    # 4. 제출 파일 생성
    print("\n[4/4] 제출 파일 생성 중...")
    output_path = Config.OUTPUT_DIR / 'submission_ensemble.csv'
    
    submission = create_submission_from_ensemble(
        predictions=predictions,
        ccf_results=ccf_results,
        sample_submission_path=Config.DATA_RAW / 'sample_submission.csv',
        output_path=output_path,
        ccf_threshold=0.3
    )
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"\n제출 파일: {output_path}")
    print(f"파일 크기: {len(submission)}개 행")
    print("=" * 60)


if __name__ == "__main__":
    main()
