"""
3-Way 앙상블: AutoGluon + Chronos + N-HiTS

최고의 3개 모델 결합
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


def load_autogluon_predictions():
    """AutoGluon 예측 로드"""
    logger.info("AutoGluon 예측 로드 중...")
    
    ag_path = Config.OUTPUT_DIR / 'submission_log' / 'autogluon_20251123_121606' / 'submission_autogluon.csv'
    
    if not ag_path.exists():
        logger.error(f"AutoGluon 파일 없음: {ag_path}")
        return None
    
    df = pd.read_csv(ag_path)
    
    # 품목별 평균
    pred_dict = {}
    for idx, row in df.iterrows():
        if row['value'] > 0:
            item = row['following_item_id']
            if item not in pred_dict:
                pred_dict[item] = []
            pred_dict[item].append(row['value'])
    
    avg_pred = {item: np.mean(values) for item, values in pred_dict.items()}
    logger.info(f"✅ AutoGluon: {len(avg_pred)}개 품목")
    return avg_pred


def load_chronos_predictions():
    """Chronos 예측 로드"""
    logger.info("Chronos 예측 로드 중...")
    
    chronos_path = Config.OUTPUT_DIR / 'submission_log' / '20251123_105903' / 'submission_chronos_large.csv'
    
    if not chronos_path.exists():
        logger.error(f"Chronos 파일 없음: {chronos_path}")
        return None
    
    df = pd.read_csv(chronos_path)
    
    # 품목별 평균
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
            max_steps=500,  # 더 많은 학습
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


def create_3way_ensemble(ag_pred, chronos_pred, nhits_pred, weights=(0.5, 0.3, 0.2)):
    """3-way 앙상블 생성"""
    w_ag, w_chronos, w_nhits = weights
    logger.info(f"앙상블 가중치: AutoGluon {w_ag:.1%}, Chronos {w_chronos:.1%}, N-HiTS {w_nhits:.1%}")
    
    # 모든 품목
    all_items = set()
    if ag_pred:
        all_items.update(ag_pred.keys())
    if chronos_pred:
        all_items.update(chronos_pred.keys())
    if nhits_pred:
        all_items.update(nhits_pred.keys())
    
    ensemble_pred = {}
    
    for item in all_items:
        preds = []
        ws = []
        
        if ag_pred and item in ag_pred:
            preds.append(ag_pred[item])
            ws.append(w_ag)
        
        if chronos_pred and item in chronos_pred:
            preds.append(chronos_pred[item])
            ws.append(w_chronos)
        
        if nhits_pred and item in nhits_pred:
            preds.append(nhits_pred[item])
            ws.append(w_nhits)
        
        if preds and len(preds) > 0:
            total_w = sum(ws)
            if total_w > 0:
                ensemble_pred[item] = sum(p * w for p, w in zip(preds, ws)) / total_w
            else:
                ensemble_pred[item] = np.mean(preds)
    
    logger.info(f"✅ 앙상블: {len(ensemble_pred)}개 품목")
    return ensemble_pred


def create_submission(predictions, output_path, confidence_threshold=0):
    """제출 파일 생성"""
    sample_submission = pd.read_csv(Config.DATA_RAW / 'sample_submission.csv')
    
    values = []
    for idx, row in sample_submission.iterrows():
        following_item = row['following_item_id']
        pred = predictions.get(following_item, 0.0)
        
        # 신뢰도 필터링
        if pred >= confidence_threshold:
            values.append(pred)
        else:
            values.append(0.0)
    
    sample_submission['value'] = values
    sample_submission.to_csv(output_path, index=False)
    
    # 통계
    n_nonzero = (sample_submission['value'] > 0).sum()
    logger.info(f"  비영 예측: {n_nonzero}개 ({n_nonzero/len(sample_submission)*100:.1f}%)")
    logger.info(f"  평균: {sample_submission['value'].mean():.2f}")
    logger.info(f"  최대: {sample_submission['value'].max():.2f}")
    
    return sample_submission


def main():
    print("=" * 60)
    print("3-Way 앙상블: AutoGluon + Chronos + N-HiTS")
    print("=" * 60)
    
    # 1. 모델 예측 로드/생성
    print("\n[1/2] 모델 예측 로드 중...")
    ag_pred = load_autogluon_predictions()
    chronos_pred = load_chronos_predictions()
    nhits_pred = generate_nhits_predictions()
    
    # 2. 다양한 앙상블 조합
    print("\n[2/2] 앙상블 생성 중...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.OUTPUT_DIR / 'submission_log' / f'3way_ensemble_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 앙상블 설정
    configs = [
        ("ag50_ch30_nh20", (0.5, 0.3, 0.2), 0),
        ("ag50_ch30_nh20_conf100k", (0.5, 0.3, 0.2), 100000),
        ("ag40_ch40_nh20", (0.4, 0.4, 0.2), 0),
        ("ag40_ch40_nh20_conf100k", (0.4, 0.4, 0.2), 100000),
        ("ag60_ch25_nh15", (0.6, 0.25, 0.15), 0),
        ("ag60_ch25_nh15_conf100k", (0.6, 0.25, 0.15), 100000),
    ]
    
    results = []
    
    for name, weights, conf_threshold in configs:
        print(f"\n--- {name} ---")
        ensemble_pred = create_3way_ensemble(ag_pred, chronos_pred, nhits_pred, weights)
        
        output_path = output_dir / f'submission_{name}.csv'
        submission = create_submission(ensemble_pred, output_path, conf_threshold)
        
        results.append({
            'name': name,
            'ag_weight': weights[0],
            'chronos_weight': weights[1],
            'nhits_weight': weights[2],
            'confidence': conf_threshold,
            'filename': output_path.name,
            'nonzero': (submission['value'] > 0).sum(),
            'mean': submission['value'].mean(),
            'max': submission['value'].max()
        })
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df[['name', 'nonzero', 'mean']].to_string(index=False))
    
    summary_path = output_dir / '3way_ensemble_summary.csv'
    results_df.to_csv(summary_path, index=False)
    
    print(f"\n저장 위치: {output_dir}")
    print("=" * 60)
    
    print("\n🎯 추천 제출 순서:")
    print("1. ag50_ch30_nh20_conf100k (균형 + 필터링)")
    print("2. ag40_ch40_nh20 (Chronos 강화)")
    print("3. ag60_ch25_nh15_conf100k (AutoGluon 우위 + 필터링)")


if __name__ == "__main__":
    main()
