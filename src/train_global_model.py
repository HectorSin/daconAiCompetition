"""
Global Model 학습 스크립트

100개 품목 데이터를 하나의 DataFrame으로 통합하여 단일 Global Model을 학습합니다.
이는 샘플 부족 품목의 성능을 크게 향상시키는 Kaggle 우승 솔루션의 정석입니다.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.cross_validation import cross_validate_model
from src.utils.experiment_logger import log_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_global_dataset(df_wide: pd.DataFrame, 
                          ccf_results: pd.DataFrame,
                          lag_features: list = [1, 2, 3, 6],
                          rolling_windows: list = [3, 6]) -> pd.DataFrame:
    """
    100개 품목 데이터를 하나의 Global Dataset으로 통합합니다.
    
    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터 (날짜 x 품목)
    ccf_results : pd.DataFrame
        CCF 결과 (선행 품목 정보)
    lag_features : list
        생성할 lag 리스트
    rolling_windows : list
        Rolling 윈도우 크기 리스트
    
    Returns:
    --------
    pd.DataFrame
        Global dataset (모든 품목이 통합된 long format)
        컬럼: item_id, date, value, lag_1, lag_2, ..., rolling_mean_3, ...
    """
    logger.info("Global Dataset 생성 중...")
    
    all_data = []
    
    for item in df_wide.columns:
        # 품목별 데이터 추출
        item_df = pd.DataFrame({
            'date': df_wide.index,
            'item_id': item,
            'value': df_wide[item].values
        })
        
        # 1. Lag 특징 생성
        for lag in lag_features:
            item_df[f'lag_{lag}'] = item_df['value'].shift(lag)
        
        # 2. Rolling 특징 생성
        for window in rolling_windows:
            item_df[f'rolling_mean_{window}'] = item_df['value'].rolling(window=window).mean()
            item_df[f'rolling_std_{window}'] = item_df['value'].rolling(window=window).std()
            item_df[f'rolling_min_{window}'] = item_df['value'].rolling(window=window).min()
            item_df[f'rolling_max_{window}'] = item_df['value'].rolling(window=window).max()
        
        # 3. 성장률 특징
        item_df['diff_1'] = item_df['value'].diff(1)
        item_df['pct_change_1'] = item_df['value'].pct_change(1) * 100
        
        # 4. 날짜 특징
        item_df['month'] = pd.to_datetime(item_df['date']).dt.month
        item_df['quarter'] = pd.to_datetime(item_df['date']).dt.quarter
        item_df['year'] = pd.to_datetime(item_df['date']).dt.year
        item_df['month_sin'] = np.sin(2 * np.pi * item_df['month'] / 12)
        item_df['month_cos'] = np.cos(2 * np.pi * item_df['month'] / 12)
        
        # 5. 선행 품목 특징 (상위 3개만)
        leading_pairs = ccf_results[
            (ccf_results['item_y'] == item) &
            (ccf_results['abs_ccf'] >= 0.5)
        ].nlargest(3, 'abs_ccf')
        
        for idx, row in leading_pairs.iterrows():
            leading_item = row['item_x']
            optimal_lag = int(row['optimal_lag'])
            
            if leading_item in df_wide.columns:
                # 선행 품목의 값을 최적 시차만큼 shift
                col_name = f'leading_{leading_item}_lag_{optimal_lag}'
                item_df[col_name] = df_wide[leading_item].shift(optimal_lag).values
        
        all_data.append(item_df)
    
    # 모든 품목 데이터 통합
    global_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Global Dataset 생성 완료: {global_df.shape}")
    logger.info(f"  - 총 행 수: {len(global_df):,} (43개월 × {len(df_wide.columns)}개 품목)")
    logger.info(f"  - 총 특징 수: {global_df.shape[1] - 3} (item_id, date, value 제외)")
    logger.info(f"  - 품목 수: {global_df['item_id'].nunique()}")
    
    return global_df


def prepare_global_train_data(global_df: pd.DataFrame, 
                               scale_per_item: bool = True) -> tuple:
    """
    Global Dataset을 학습용으로 준비합니다.
    
    Parameters:
    -----------
    global_df : pd.DataFrame
        Global dataset
    scale_per_item : bool
        품목별로 스케일링할지 여부
    
    Returns:
    --------
    tuple
        (X, y, item_ids, feature_names, scaler)
    """
    logger.info("학습 데이터 준비 중...")
    
    # 결측값 제거 전 상태 확인
    logger.info(f"결측값 제거 전: {len(global_df):,}행")
    logger.info(f"  - 품목당 평균 행 수: {len(global_df) / global_df['item_id'].nunique():.1f}")
    
    # 결측값 통계
    null_counts = global_df.isnull().sum()
    if null_counts.sum() > 0:
        logger.info(f"결측값이 있는 컬럼 (상위 5개):")
        for col, count in null_counts.nlargest(5).items():
            logger.info(f"  - {col}: {count:,}개 ({count/len(global_df)*100:.1f}%)")
    
    # 핵심 특징과 선행 특징 분리
    leading_cols = [col for col in global_df.columns if col.startswith('leading_')]
    core_cols = [col for col in global_df.columns if col not in leading_cols and col not in ['date', 'item_id', 'value']]
    
    logger.info(f"\n특징 분류:")
    logger.info(f"  - 핵심 특징: {len(core_cols)}개 (lag, rolling, date 등)")
    logger.info(f"  - 선행 특징: {len(leading_cols)}개 (leading_*)")
    
    # 선택적 NaN 처리
    # 1. 핵심 특징에서 NaN이 있는 행만 제거
    df_clean = global_df.dropna(subset=core_cols)
    logger.info(f"\n핵심 특징 NaN 제거 후: {len(df_clean):,}행 (제거: {len(global_df) - len(df_clean):,}행)")
    
    # 2. 선행 특징의 NaN은 0으로 채우기 (해당 품목에 선행 관계가 없음을 의미)
    if leading_cols:
        df_clean[leading_cols] = df_clean[leading_cols].fillna(0)
        logger.info(f"선행 특징 NaN을 0으로 채움 (선행 관계 없음을 의미)")
    
    
    logger.info(f"\n최종 데이터: {len(df_clean):,}행")
    if len(df_clean) > 0:
        logger.info(f"  - 품목당 평균 행 수: {len(df_clean) / df_clean['item_id'].nunique():.1f}")
    else:
        raise ValueError(
            "모든 데이터가 제거되었습니다!\n"
            "  - 원인: 핵심 특징(lag, rolling)에 NaN이 너무 많음\n"
            "  - 해결: create_global_dataset()의 lag_features 또는 rolling_windows를 줄여보세요."
        )

    
    # item_id를 categorical로 인코딩
    label_encoder = LabelEncoder()
    df_clean['item_id_encoded'] = label_encoder.fit_transform(df_clean['item_id'])
    
    # 특징과 타겟 분리
    exclude_cols = ['date', 'item_id', 'value']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols].copy()
    y = df_clean['value'].copy()
    item_ids = df_clean['item_id'].copy()
    
    # 품목별 스케일링 (선택적)
    scaler = None
    if scale_per_item:
        logger.info("품목별 StandardScaler 적용 중...")
        scaler = {}
        X_scaled = X.copy()
        
        for item in df_clean['item_id'].unique():
            item_mask = df_clean['item_id'] == item
            item_scaler = StandardScaler()
            
            # item_id_encoded는 스케일링하지 않음
            numeric_cols = [col for col in feature_cols if col != 'item_id_encoded']
            X_scaled.loc[item_mask, numeric_cols] = item_scaler.fit_transform(
                X.loc[item_mask, numeric_cols]
            )
            scaler[item] = item_scaler
        
        X = X_scaled
    
    logger.info(f"학습 데이터 준비 완료:")
    logger.info(f"  - X shape: {X.shape}")
    logger.info(f"  - y shape: {y.shape}")
    logger.info(f"  - 전체 특징 수: {len(feature_cols)}")
    logger.info(f"  - item_id_encoded 포함 여부: {'item_id_encoded' in X.columns}")
    logger.info(f"  - X 컬럼: {X.columns.tolist()[:10]}...")  # 처음 10개만
    
    return X, y, item_ids, feature_cols, scaler, label_encoder


def train_global_model(X: pd.DataFrame, 
                       y: pd.Series,
                       use_cv: bool = True,
                       n_splits: int = 3,
                       test_size: int = 3,
                       model_params: dict = None) -> dict:
    """
    Global Model을 학습합니다.
    
    Parameters:
    -----------
    X : pd.DataFrame
        특징 데이터
    y : pd.Series
        타겟 데이터
    use_cv : bool
        교차 검증 사용 여부
    n_splits : int
        CV fold 수
    test_size : int
        각 fold의 validation 크기
    model_params : dict
        모델 파라미터
    
    Returns:
    --------
    dict
        학습 결과 (model, cv_results, metrics)
    """
    if model_params is None:
        model_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 8,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': Config.SEED,
            'verbose': -1,
            'categorical_feature': ['item_id_encoded']  # fit()에는 'name:' 접두사 없이 전달
        }
    
    logger.info("Global Model 학습 시작...")
    logger.info(f"모델 파라미터: {model_params}")
    
    if use_cv:
        # Forward Chaining CV로 학습
        logger.info(f"Forward Chaining CV 사용 (n_splits={n_splits}, test_size={test_size})")
        
        cv_results = cross_validate_model(
            model_class=LGBMRegressor,
            X=X,
            y=y,
            n_splits=n_splits,
            test_size=test_size,
            model_params=model_params,
            verbose=True
        )
        
        # 전체 데이터로 최종 모델 학습
        logger.info("\n전체 데이터로 최종 모델 학습 중...")
        
        # categorical_feature를 model_params에서 분리
        fit_params = {}
        final_model_params = model_params.copy()
        if 'categorical_feature' in final_model_params:
            fit_params['categorical_feature'] = final_model_params.pop('categorical_feature')
        
        final_model = LGBMRegressor(**final_model_params)
        final_model.fit(X, y, **fit_params)
        
        return {
            'model': final_model,
            'cv_results': cv_results,
            'mean_val_rmse': cv_results['mean_val_rmse'],
            'mean_val_mae': cv_results['mean_val_mae'],
            'std_val_rmse': cv_results['std_val_rmse'],
            'std_val_mae': cv_results['std_val_mae']
        }
    else:
        # CV 없이 바로 학습
        # categorical_feature를 model_params에서 분리
        fit_params = {}
        model_params_copy = model_params.copy()
        if 'categorical_feature' in model_params_copy:
            fit_params['categorical_feature'] = model_params_copy.pop('categorical_feature')
        
        model = LGBMRegressor(**model_params_copy)
        model.fit(X, y, **fit_params)
        
        # Train 성능 평가
        y_pred = model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_mae = mean_absolute_error(y, y_pred)
        
        logger.info(f"Train RMSE: {train_rmse:,.2f}")
        logger.info(f"Train MAE: {train_mae:,.2f}")
        
        return {
            'model': model,
            'cv_results': None,
            'train_rmse': train_rmse,
            'train_mae': train_mae
        }


def main():
    print("=" * 60)
    print("Global Model 학습")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[1/5] 데이터 로드 중...")
    df_raw = pd.read_csv(Config.DATA_RAW / 'train.csv')
    df_raw['date'] = pd.to_datetime(df_raw[['year', 'month']].assign(day=1))
    
    df_agg = df_raw.groupby(['date', 'item_id']).agg({
        'value': 'sum'
    }).reset_index()
    
    df_wide = df_agg.pivot(index='date', columns='item_id', values='value').fillna(0)
    print(f"데이터 shape: {df_wide.shape} (날짜 x 품목)")
    
    # 2. CCF 결과 로드
    print("\n[2/5] CCF 결과 로드 중...")
    ccf_results = pd.read_csv(Config.DATA_PROCESSED / 'ccf_results.csv')
    print(f"CCF 결과: {len(ccf_results)}개 쌍")
    
    # 3. Global Dataset 생성
    print("\n[3/5] Global Dataset 생성 중...")
    global_df = create_global_dataset(
        df_wide, 
        ccf_results,
        lag_features=[1, 2, 3, 6],
        rolling_windows=[3, 6]
    )
    
    # 4. 학습 데이터 준비
    print("\n[4/5] 학습 데이터 준비 중...")
    X, y, item_ids, feature_names, scaler, label_encoder = prepare_global_train_data(
        global_df,
        scale_per_item=False  # Global Model에서는 품목별 스케일링 비활성화
    )
    
    # 5. Global Model 학습
    print("\n[5/5] Global Model 학습 중...")
    
    # 데이터 크기에 따라 CV 파라미터 조정
    n_samples = len(X)
    print(f"학습 가능한 샘플 수: {n_samples:,}")
    
    # CV 파라미터 자동 조정
    if n_samples >= 30:
        n_splits, test_size = 3, 3
        use_cv = True
        print(f"CV 설정: n_splits={n_splits}, test_size={test_size}")
    elif n_samples >= 20:
        n_splits, test_size = 2, 2
        use_cv = True
        print(f"CV 설정 (데이터 부족으로 축소): n_splits={n_splits}, test_size={test_size}")
    else:
        use_cv = False
        print(f"⚠️ 샘플 수가 부족하여 CV를 건너뜁니다 (최소 20개 필요, 현재 {n_samples}개)")
    
    result = train_global_model(
        X, y,
        use_cv=use_cv,
        n_splits=n_splits if use_cv else 3,
        test_size=test_size if use_cv else 3
    )
    
    # 6. 모델 저장
    print("\n모델 저장 중...")
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = Config.MODELS_DIR / 'global_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': result['model'],
            'feature_names': feature_names,
            'label_encoder': label_encoder,
            'scaler': scaler,
            'cv_results': result.get('cv_results'),
            'metrics': {
                'mean_val_rmse': result.get('mean_val_rmse'),
                'mean_val_mae': result.get('mean_val_mae'),
                'std_val_rmse': result.get('std_val_rmse'),
                'std_val_mae': result.get('std_val_mae')
            }
        }, f)
    
    print(f"모델 저장 완료: {model_path}")
    
    # 7. 실험 로깅
    if result.get('cv_results'):
        log_experiment(
            experiment_name='global_model_baseline',
            model_type='LightGBM_Global',
            params={
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 8
            },
            cv_score=result['mean_val_rmse'],
            cv_std=result['std_val_rmse'],
            notes='100개 품목 통합 Global Model (item_id categorical)'
        )
    
    # 요약
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    if result.get('cv_results'):
        print(f"평균 Val RMSE: {result['mean_val_rmse']:,.2f} (±{result['std_val_rmse']:,.2f})")
        print(f"평균 Val MAE: {result['mean_val_mae']:,.2f} (±{result['std_val_mae']:,.2f})")
    print(f"특징 개수: {len(feature_names)}")
    print(f"학습 샘플 수: {len(X):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
