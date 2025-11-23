"""
AutoGluon을 사용한 시계열 예측

가장 강력한 AutoML 솔루션
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data():
    """AutoGluon 형식으로 데이터 준비"""
    logger.info("데이터 준비 중...")
    
    df_raw = pd.read_csv(Config.DATA_RAW / 'train.csv')
    df_raw['timestamp'] = pd.to_datetime(df_raw[['year', 'month']].assign(day=1))
    
    # 품목별 시계열
    df_agg = df_raw.groupby(['timestamp', 'item_id']).agg({
        'value': 'sum'
    }).reset_index()
    
    # AutoGluon 형식: item_id, timestamp, target
    df_agg = df_agg.rename(columns={'value': 'target'})
    
    logger.info(f"데이터: {df_agg.shape}")
    logger.info(f"품목 수: {df_agg['item_id'].nunique()}")
    logger.info(f"기간: {df_agg['timestamp'].min()} ~ {df_agg['timestamp'].max()}")
    
    return df_agg


def train_autogluon(train_data, time_limit=7200):
    """AutoGluon 학습"""
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    
    logger.info("AutoGluon 학습 시작...")
    
    # TimeSeriesDataFrame 변환
    ts_df = TimeSeriesDataFrame.from_data_frame(
        train_data,
        id_column='item_id',
        timestamp_column='timestamp'
    )
    
    logger.info(f"TimeSeriesDataFrame: {ts_df.num_items} items")
    
    # Predictor 생성
    predictor = TimeSeriesPredictor(
        prediction_length=1,
        path=str(Config.MODELS_DIR / 'autogluon'),
        target='target',
        eval_metric='MAPE',
        freq='MS',  # 월초 (Month Start)
        verbosity=2
    )
    
    # 학습
    logger.info(f"학습 시작 (제한 시간: {time_limit}초)...")
    predictor.fit(
        train_data=ts_df,
        presets='best_quality',  # 최고 품질
        time_limit=time_limit,
        num_val_windows=3,  # CV
        enable_ensemble=True
    )
    
    logger.info("✅ 학습 완료")
    
    return predictor


def generate_predictions(predictor, train_data):
    """예측 생성"""
    from autogluon.timeseries import TimeSeriesDataFrame
    
    logger.info("예측 생성 중...")
    
    # TimeSeriesDataFrame 변환
    ts_df = TimeSeriesDataFrame.from_data_frame(
        train_data,
        id_column='item_id',
        timestamp_column='timestamp'
    )
    
    # 예측
    predictions = predictor.predict(ts_df)
    
    # 딕셔너리로 변환
    pred_dict = {}
    for item_id in predictions.item_ids:
        pred_value = predictions.loc[item_id]['mean'].iloc[0]
        pred_dict[item_id] = max(0, pred_value)
    
    logger.info(f"✅ {len(pred_dict)}개 품목 예측 완료")
    return pred_dict


def create_submission(predictions, output_path):
    """제출 파일 생성"""
    sample_submission = pd.read_csv(Config.DATA_RAW / 'sample_submission.csv')
    
    values = []
    for idx, row in sample_submission.iterrows():
        following_item = row['following_item_id']
        pred = predictions.get(following_item, 0.0)
        values.append(pred)
    
    sample_submission['value'] = values
    sample_submission.to_csv(output_path, index=False)
    
    # 통계
    n_nonzero = (sample_submission['value'] > 0).sum()
    logger.info(f"제출 파일 생성 완료:")
    logger.info(f"  비영 예측: {n_nonzero}개 ({n_nonzero/len(sample_submission)*100:.1f}%)")
    logger.info(f"  평균: {sample_submission['value'].mean():.2f}")
    logger.info(f"  최대: {sample_submission['value'].max():.2f}")
    
    return sample_submission


def main():
    print("=" * 60)
    print("AutoGluon 시계열 예측")
    print("=" * 60)
    
    # 1. 데이터 준비
    print("\n[1/3] 데이터 준비...")
    train_data = prepare_data()
    
    # 2. AutoGluon 학습
    print("\n[2/3] AutoGluon 학습...")
    predictor = train_autogluon(train_data, time_limit=7200)  # 2시간
    
    # 3. 예측 및 제출 파일 생성
    print("\n[3/3] 예측 생성...")
    predictions = generate_predictions(predictor, train_data)
    
    # 제출 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.OUTPUT_DIR / 'submission_log' / f'autogluon_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'submission_autogluon.csv'
    create_submission(predictions, output_path)
    
    print(f"\n저장 위치: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
