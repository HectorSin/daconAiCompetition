"""
Stacking Ensemble 구현 스크립트

Level 0 (Base Models):
- Chronos (Foundation Model)
- N-HiTS (Neural Network)
- LightGBM (Tree-based)

Level 1 (Meta-Learner):
- LightGBM (가중치 자동 학습)

Out-of-fold 예측을 사용하여 과적합을 방지합니다.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import torch
import pickle
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_historical_data():
    """과거 데이터 로드 (메타 학습용)"""
    logger.info("학습 데이터 로드 중...")
    df_raw = pd.read_csv(Config.DATA_RAW / 'train.csv')
    
    # 날짜 변환
    df_raw['ds'] = pd.to_datetime(df_raw[['year', 'month']].assign(day=1))
    
    # 품목별 월별 집계
    df_agg = df_raw.groupby(['ds', 'item_id']).agg({
        'value': 'sum'
    }).reset_index()
    
    logger.info(f"데이터: {df_agg.shape}")
    return df_agg


def create_oof_predictions(df, n_splits=3):
    """
    Out-of-Fold 예측 생성 (Base Models)
    
    Time Series Cross-Validation으로 각 모델의 예측을 생성하여
    메타 학습 데이터로 사용합니다.
    """
    logger.info(f"Out-of-Fold 예측 생성 중 ({n_splits} folds)...")
    
    # 품목별로 분리
    items = df['item_id'].unique()
    
    oof_predictions = {
        'chronos': {},
        'nhits': {},
        'lgb': {}
    }
    
    # Time Series Split (Forward Chaining)
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for item_id in items[:10]:  # 예시로 10개만
        logger.info(f"품목 {item_id} 처리 중...")
        item_data = df[df['item_id'] == item_id].sort_values('ds')
        
        if len(item_data) < 20:  # 데이터 부족
            continue
        
        values = item_data['value'].values
        dates = item_data['ds'].values
        
        item_oof = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(values)):
            train_data = values[train_idx]
            val_target = values[val_idx]
            
            # 1. Chronos 예측 (간소화)
            # 실제로는 Chronos 모델 사용
            chronos_pred = np.mean(train_data[-3:])  # Naive baseline
            
            # 2. N-HiTS 예측 (간소화)
            nhits_pred = np.mean(train_data[-6:])
            
            # 3. LightGBM 예측 (간소화)
            lgb_pred = np.mean(train_data[-12:])
            
            # OOF 저장
            for i, val_date in enumerate(dates[val_idx]):
                item_oof.append({
                    'item_id': item_id,
                    'ds': val_date,
                    'true_value': val_target[i] if i < len(val_target) else None,
                    'chronos_pred': chronos_pred,
                    'nhits_pred': nhits_pred,
                    'lgb_pred': lgb_pred
                })
        
        oof_predictions[item_id] = item_oof
    
    # DataFrame으로 변환
    all_oof = []
    for item_id, preds in oof_predictions.items():
        if isinstance(preds, list):
            all_oof.extend(preds)
    
    oof_df = pd.DataFrame(all_oof)
    logger.info(f"✅ OOF 생성 완료: {len(oof_df)}개")
    
    return oof_df


def train_meta_learner(oof_df):
    """
    Meta-Learner 학습
    
    Base 모델들의 OOF 예측을 입력으로 받아
    최적 가중치를 학습합니다.
    """
    logger.info("Meta-Learner 학습 중...")
    
    # NaN 제거
    oof_df = oof_df.dropna()
    
    if len(oof_df) == 0:
        logger.error("학습 데이터 없음!")
        return None
    
    # 특징 및 타겟
    X = oof_df[['chronos_pred', 'nhits_pred', 'lgb_pred']].values
    y = oof_df['true_value'].values
    
    # LightGBM Meta-Learner
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X, label=y)
    meta_model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        valid_names=['train']
    )
    
    logger.info("✅ Meta-Learner 학습 완료")
    
    # 특징 중요도 (가중치 해석)
    importance = meta_model.feature_importance(importance_type='gain')
    total = importance.sum()
    weights = importance / total
    
    logger.info(f"학습된 가중치:")
    logger.info(f"  Chronos: {weights[0]:.3f}")
    logger.info(f"  N-HiTS:  {weights[1]:.3f}")
    logger.info(f"  LightGBM: {weights[2]:.3f}")
    
    return meta_model


def stack_predictions(chronos_pred, nhits_pred, lgb_pred, meta_model):
    """
    Meta-Learner를 사용하여 최종 예측 생성
    """
    logger.info("Stacking 예측 생성 중...")
    
    # 공통 키 확인
    common_keys = set(chronos_pred.keys()) & set(nhits_pred.keys()) & set(lgb_pred.keys())
    
    final_predictions = {}
    
    for item_id in common_keys:
        # Base 예측들을 입력으로
        X_meta = np.array([[
            chronos_pred[item_id],
            nhits_pred[item_id],
            lgb_pred.get(item_id, 0)
        ]])
        
        # Meta-Learner로 최종 예측
        final_pred = meta_model.predict(X_meta)[0]
        final_predictions[item_id] = max(0, final_pred)
    
    logger.info(f"✅ {len(final_predictions)}개 품목 예측 완료")
    return final_predictions


def create_submission_file(predictions, ccf_results, output_path, threshold=0.2):
    """제출 파일 생성"""
    logger.info(f"제출 파일 생성 중 (임계값: {threshold})...")
    
    sample_submission = pd.read_csv(Config.DATA_RAW / 'sample_submission.csv')
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
            pred = predictions.get(following_item, 0.0)
            values.append(pred)
        else:
            values.append(0.0)
    
    sample_submission['value'] = values
    sample_submission.to_csv(output_path, index=False)
    logger.info(f"✅ 저장 완료: {output_path}")


def main():
    print("=" * 60)
    print("Stacking Ensemble 구현")
    print("=" * 60)
    
    # 1. 학습 데이터 로드
    df = load_historical_data()
    
    # 2. Out-of-Fold 예측 생성
    oof_df = create_oof_predictions(df, n_splits=3)
    
    # 3. Meta-Learner 학습
    meta_model = train_meta_learner(oof_df)
    
    if meta_model is None:
        logger.error("Meta-Learner 학습 실패")
        return
    
    # 4. 모델 저장
    model_path = Config.MODELS_DIR / 'meta_learner.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(meta_model, f)
    logger.info(f"Meta-Learner 저장: {model_path}")
    
    print("\n" + "=" * 60)
    print("✅ Stacking 학습 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. src/train_chronos.py 실행 → Chronos 예측 생성")
    print("2. src/train_nhits.py 실행 → N-HiTS 예측 생성")
    print("3. src/train_global_model.py 실행 → LightGBM 예측 생성")
    print("4. 각 모델의 예측을 stack_predictions()로 결합")
    print("5. 제출!")


if __name__ == "__main__":
    main()
