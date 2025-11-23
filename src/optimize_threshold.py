"""
CCF 임계값 최적화 스크립트

다양한 CCF 임계값(0.2 ~ 0.4)으로 제출 파일을 생성하여
최적의 임계값을 찾습니다.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.ensemble_local_global import create_ensemble_predictions, create_submission_from_ensemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("CCF 임계값 최적화")
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
    
    # 3. 앙상블 예측 (한 번만 수행)
    print("\n[3/4] 앙상블 예측 생성 중...")
    predictions = create_ensemble_predictions(
        df_wide=df_wide,
        ccf_results=ccf_results,
        local_models_dir=Config.MODELS_DIR,
        global_model_path=Config.MODELS_DIR / 'global_model.pkl',
        local_weight=0.7
    )
    
    # 4. 다양한 임계값으로 제출 파일 생성
    print("\n[4/4] 다양한 임계값 테스트 중...")
    
    # 출력 디렉토리 생성 (현재 시간)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.OUTPUT_DIR / 'submission_log' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리: {output_dir}")
    
    thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # 더 높은 임계값 테스트
    
    for threshold in thresholds:
        print(f"\n--- 임계값: {threshold} ---")
        output_path = output_dir / f'submission_ensemble_ccf_{threshold}.csv'
        
        submission = create_submission_from_ensemble(
            predictions=predictions,
            ccf_results=ccf_results,
            sample_submission_path=Config.DATA_RAW / 'sample_submission.csv',
            output_path=output_path,
            ccf_threshold=threshold
        )
        
        # 통계 출력
        n_nonzero = (submission['value'] > 0).sum()
        print(f"공행성 쌍: {n_nonzero}개 ({n_nonzero/len(submission)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
