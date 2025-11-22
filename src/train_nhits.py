"""
N-HiTS 모델 학습 및 예측 스크립트

NeuralForecast 라이브러리의 N-HiTS 모델을 사용하여
시계열 예측을 수행하고 제출 파일을 생성합니다.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import torch
from datetime import datetime
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_data():
    """
    데이터를 로드하고 NeuralForecast 형식으로 변환합니다.
    Format: unique_id, ds, y
    """
    logger.info("데이터 로드 중...")
    df_raw = pd.read_csv(Config.DATA_RAW / 'train.csv')
    
    # 날짜 변환
    df_raw['ds'] = pd.to_datetime(df_raw[['year', 'month']].assign(day=1))
    
    # 집계 (월별, 품목별)
    df_agg = df_raw.groupby(['ds', 'item_id']).agg({
        'value': 'sum'
    }).reset_index()
    
    # 컬럼명 변경 (NeuralForecast 요구사항)
    df_agg = df_agg.rename(columns={
        'item_id': 'unique_id',
        'value': 'y'
    })
    
    logger.info(f"데이터 전처리 완료: {df_agg.shape}")
    return df_agg


def train_nhits(df):
    """
    N-HiTS 모델을 학습합니다.
    """
    logger.info("N-HiTS 모델 설정 및 학습 중...")
    
    # 모델 설정
    # horizon=1 (다음 달 예측), input_size=12 (과거 12개월 사용)
    models = [
        NHITS(
            h=1,
            input_size=12,
            loss=MAE(),
            max_steps=500,  # 학습 단계 수
            learning_rate=1e-3,
            val_check_steps=50,
            early_stop_patience_steps=10,
            scaler_type='standard',  # 표준화
            enable_progress_bar=True
        )
    ]
    
    nf = NeuralForecast(
        models=models,
        freq='MS'  # 월초 데이터
    )
    
    nf.fit(df=df, val_size=3)
    logger.info("모델 학습 완료")
    
    return nf


def create_submission_file(predictions, ccf_results, output_path, threshold=0.2):
    """
    예측값을 기반으로 제출 파일을 생성합니다.
    """
    logger.info(f"제출 파일 생성 중 (임계값: {threshold})...")
    
    sample_submission = pd.read_csv(Config.DATA_RAW / 'sample_submission.csv')
    
    # 예측값 딕셔너리로 변환
    pred_dict = predictions.set_index('unique_id')['NHITS'].to_dict()
    
    values = []
    
    for idx, row in sample_submission.iterrows():
        leading_item = row['leading_item_id']
        following_item = row['following_item_id']
        
        # CCF 확인
        comovement = ccf_results[
            ((ccf_results['item_x'] == leading_item) & (ccf_results['item_y'] == following_item)) |
            ((ccf_results['item_x'] == following_item) & (ccf_results['item_y'] == leading_item))
        ]
        
        has_comovement = False
        if len(comovement) > 0:
            max_ccf = comovement['abs_ccf'].max()
            if max_ccf >= threshold:
                has_comovement = True
        
        if has_comovement:
            # 음수 예측값 방지
            pred = max(0, pred_dict.get(following_item, 0.0))
            values.append(pred)
        else:
            values.append(0.0)
            
        if (idx + 1) % 1000 == 0:
            logger.info(f"진행: {idx+1}/{len(sample_submission)}")
            
    sample_submission['value'] = values
    sample_submission.to_csv(output_path, index=False)
    logger.info(f"저장 완료: {output_path}")
    
    # 통계
    n_nonzero = (sample_submission['value'] > 0).sum()
    logger.info(f"공행성 쌍: {n_nonzero}개 ({n_nonzero/len(sample_submission)*100:.1f}%)")


def main():
    print("=" * 60)
    print("N-HiTS 모델 학습 및 예측")
    print("=" * 60)
    
    # 1. 데이터 준비
    df = load_and_preprocess_data()
    
    # 2. 모델 학습
    nf = train_nhits(df)
    
    # 3. 예측 (2025-08)
    logger.info("예측 생성 중...")
    forecast = nf.predict()
    forecast = forecast.reset_index()
    
    # 2025-08 데이터만 필터링 (이미 h=1이라 하나만 나옴)
    print(forecast.head())
    
    # 4. 제출 파일 생성
    ccf_results = pd.read_csv(Config.DATA_PROCESSED / 'ccf_results.csv')
    
    # 타임스탬프 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.OUTPUT_DIR / 'submission_log' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리: {output_dir}")
    
    output_path = output_dir / 'submission_nhits_0.2.csv'
    
    create_submission_file(forecast, ccf_results, output_path, threshold=0.2)
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
