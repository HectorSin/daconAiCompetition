"""
종합 앙상블 시스템 - 간소화 버전

Chronos + N-HiTS 앙상블
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


def load_chronos_predictions():
    """Chronos 예측 로드"""
    logger.info("Chronos 예측 로드 중...")
    chronos_path = Config.OUTPUT_DIR / 'submission_log' / '20251123_105903' / 'submission_chronos_large.csv'
    
    if not chronos_path.exists():
        logger.error(f"Chronos 파일 없음: {chronos_path}")
        return None
    
    df = pd.read_csv(chronos_path)
    
    # 품목별 평균 예측값
    pred_dict = {}
    for idx, row in df.iterrows():
        if row['value'] > 0:
            item = row['following_item_id']
            if item not in pred_dict:
                pred_dict[item] = []
            pred_dict[item].append(row['value'])
    
    avg_pred = {item: np.mean(values) for item, values in pred_dict.items()}
    logger.info(f"✅ Chronos: {len(avg_pred)}개 품목")
    return avg_pred


def generate_nhits_predictions():
    """N-HiTS 예측 생성"""
    logger.info("N-HiTS 예측 생성 중...")
    
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MAE
    
    # 데이터 로드
    df_raw = pd.read_csv(Config.DATA_RAW / 'train.csv')
    df_raw['ds'] = pd.to_datetime(df_raw[['year', 'month']].assign(day=1))
    
    df_agg = df_raw.groupby(['ds', 'item_id']).agg({
        'value': 'sum'
    }).reset_index()
    
    df_agg = df_agg.rename(columns={
        'item_id': 'unique_id',
        'value': 'y'
    })
    
    # N-HiTS 모델
    models = [
        NHITS(
            h=1,
            input_size=12,
            loss=MAE(),
            max_steps=300,
            learning_rate=1e-3,
            scaler_type='standard',
            enable_progress_bar=False
        )
    ]
    
    nf = NeuralForecast(models=models, freq='MS')
    nf.fit(df=df_agg, val_size=3)
    
    # 예측
    forecast = nf.predict()
    forecast = forecast.reset_index()
    
    pred_dict = forecast.set_index('unique_id')['NHITS'].to_dict()
    logger.info(f"✅ N-HiTS: {len(pred_dict)}개 품목")
    return pred_dict


def create_ensemble(chronos_pred, nhits_pred, weight_chronos=0.6):
    """앙상블 생성"""
    weight_nhits = 1.0 - weight_chronos
    logger.info(f"앙상블 가중치: Chronos {weight_chronos:.1%}, N-HiTS {weight_nhits:.1%}")
    
    # 모든 품목
    all_items = set(list(chronos_pred.keys()) + list(nhits_pred.keys()))
    
    ensemble_pred = {}
    for item in all_items:
        c_pred = chronos_pred.get(item, 0)
        n_pred = nhits_pred.get(item, 0)
        
        # 둘 다 있으면 가중 평균
        if c_pred > 0 and n_pred > 0:
            ensemble_pred[item] = weight_chronos * c_pred + weight_nhits * n_pred
        # 하나만 있으면 그것 사용
        elif c_pred > 0:
            ensemble_pred[item] = c_pred
        elif n_pred > 0:
            ensemble_pred[item] = n_pred
    
    logger.info(f"✅ 앙상블: {len(ensemble_pred)}개 품목")
    return ensemble_pred


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
    logger.info(f"  비영 예측: {n_nonzero}개 ({n_nonzero/len(sample_submission)*100:.1f}%)")
    logger.info(f"  평균: {sample_submission['value'].mean():.2f}")
    
    return sample_submission


def main():
    print("=" * 60)
    print("Chronos + N-HiTS 앙상블")
    print("=" * 60)
    
    # 1. 모델 예측 생성
    print("\n[1/2] 모델 예측 생성 중...")
    chronos_pred = load_chronos_predictions()
    nhits_pred = generate_nhits_predictions()
    
    # 2. 앙상블 조합
    print("\n[2/2] 앙상블 생성 중...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.OUTPUT_DIR / 'submission_log' / f'ensemble_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 다양한 가중치 테스트
    weights = [0.5, 0.6, 0.7, 0.8]
    
    results = []
    for w in weights:
        print(f"\n--- Chronos {w:.0%} / N-HiTS {1-w:.0%} ---")
        ensemble_pred = create_ensemble(chronos_pred, nhits_pred, w)
        
        output_path = output_dir / f'submission_chronos_{int(w*100)}_nhits_{int((1-w)*100)}.csv'
        submission = create_submission(ensemble_pred, output_path)
        
        results.append({
            'chronos_weight': w,
            'nhits_weight': 1-w,
            'filename': output_path.name,
            'nonzero': (submission['value'] > 0).sum()
        })
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print(f"\n저장 위치: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
