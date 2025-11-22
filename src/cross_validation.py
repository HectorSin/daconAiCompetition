"""
시계열 교차 검증 시스템 (Forward Chaining CV)

Forward Chaining 방식으로 Data Leakage를 최소화합니다.
Train 세트가 점진적으로 확장되며, 항상 과거 데이터로 미래를 예측합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ForwardChainingCV:
    """
    Forward Chaining 교차 검증 클래스
    
    예시 (n_splits=3, test_size=3):
    - Fold 1: Train [1:20] → Val [21:23]
    - Fold 2: Train [1:23] → Val [24:26]
    - Fold 3: Train [1:26] → Val [27:29]
    
    Train 세트가 점진적으로 확장되어 Data Leakage 위험이 최소화됩니다.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 3):
        """
        Args:
            n_splits: CV fold 수 (기본값: 5)
            test_size: 각 fold의 validation 크기 (기본값: 3개월)
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        시계열 데이터를 Forward Chaining 방식으로 분할
        
        Args:
            X: 입력 데이터 (시간 순서로 정렬되어 있어야 함)
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_samples = len(X)
        splits = []
        
        # 최소 train 크기 계산
        min_train_size = n_samples - (self.n_splits * self.test_size)
        min_required_samples = 10 + self.n_splits * self.test_size
        
        if min_train_size < 10:
            raise ValueError(
                f"데이터가 부족합니다.\n"
                f"  - 현재 샘플 수: {n_samples}개\n"
                f"  - 필요 샘플 수: {min_required_samples}개 (최소 train={10} + n_splits={self.n_splits} × test_size={self.test_size})\n"
                f"  - 해결 방법: n_splits 또는 test_size를 줄이거나, use_cv=False로 설정하세요."
            )
        
        for i in range(self.n_splits):
            # Validation 끝 인덱스
            val_end = n_samples - (self.n_splits - i - 1) * self.test_size
            val_start = val_end - self.test_size
            
            # Train은 처음부터 validation 시작 전까지
            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)
            
            splits.append((train_idx, val_idx))
        
        return splits
    
    def validate_no_leakage(self, train_idx: np.ndarray, val_idx: np.ndarray) -> bool:
        """
        Data Leakage 검증: Train의 모든 인덱스가 Val보다 작은지 확인
        
        Args:
            train_idx: Train 인덱스
            val_idx: Validation 인덱스
        
        Returns:
            True if no leakage, False otherwise
        """
        max_train_idx = train_idx.max()
        min_val_idx = val_idx.min()
        
        if max_train_idx >= min_val_idx:
            print(f"⚠️ Data Leakage 감지! Max Train idx: {max_train_idx}, Min Val idx: {min_val_idx}")
            return False
        return True
    
    def get_split_info(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        각 Fold의 분할 정보를 DataFrame으로 반환
        
        Args:
            X: 입력 데이터
        
        Returns:
            Split 정보 DataFrame
        """
        info = []
        for fold, (train_idx, val_idx) in enumerate(self.split(X)):
            info.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_ratio': len(train_idx) / len(X),
                'val_ratio': len(val_idx) / len(X),
                'no_leakage': self.validate_no_leakage(train_idx, val_idx)
            })
        return pd.DataFrame(info)


def cross_validate_model(
    model_class,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 3,
    model_params: Dict[str, Any] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Forward Chaining CV로 모델 성능 평가
    
    Args:
        model_class: 모델 클래스 (예: LGBMRegressor)
        X: 특징 데이터
        y: 타겟 데이터
        n_splits: CV fold 수
        test_size: 각 fold의 validation 크기
        model_params: 모델 파라미터
        verbose: 진행 상황 출력 여부
    
    Returns:
        CV 결과 딕셔너리 (scores, oof_predictions, models)
    """
    if model_params is None:
        model_params = {}
    
    cv = ForwardChainingCV(n_splits=n_splits, test_size=test_size)
    
    # 결과 저장
    cv_scores = {
        'fold': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_mae': [],
        'val_mae': []
    }
    
    oof_predictions = np.zeros(len(y))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{n_splits}")
            print(f"{'='*50}")
        
        # Data Leakage 검증
        if not cv.validate_no_leakage(train_idx, val_idx):
            raise ValueError(f"Fold {fold + 1}에서 Data Leakage 감지!")
        
        # Train/Val 분할
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if verbose:
            print(f"Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
        
        # categorical_feature를 model_params에서 분리
        fit_params = {}
        model_params_copy = model_params.copy()
        if 'categorical_feature' in model_params_copy:
            fit_params['categorical_feature'] = model_params_copy.pop('categorical_feature')
        
        # 모델 학습
        model = model_class(**model_params_copy)
        model.fit(X_train, y_train, **fit_params)
        
        # 예측
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # 메트릭 계산
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        # 결과 저장
        cv_scores['fold'].append(fold + 1)
        cv_scores['train_rmse'].append(train_rmse)
        cv_scores['val_rmse'].append(val_rmse)
        cv_scores['train_mae'].append(train_mae)
        cv_scores['val_mae'].append(val_mae)
        
        oof_predictions[val_idx] = val_pred
        models.append(model)
        
        if verbose:
            print(f"Train RMSE: {train_rmse:,.2f} | Val RMSE: {val_rmse:,.2f}")
            print(f"Train MAE:  {train_mae:,.2f} | Val MAE:  {val_mae:,.2f}")
    
    # 전체 평균 계산
    cv_scores_df = pd.DataFrame(cv_scores)
    
    if verbose:
        print(f"\n{'='*50}")
        print("CV 결과 요약")
        print(f"{'='*50}")
        print(f"평균 Train RMSE: {cv_scores_df['train_rmse'].mean():,.2f} (±{cv_scores_df['train_rmse'].std():,.2f})")
        print(f"평균 Val RMSE:   {cv_scores_df['val_rmse'].mean():,.2f} (±{cv_scores_df['val_rmse'].std():,.2f})")
        print(f"평균 Train MAE:  {cv_scores_df['train_mae'].mean():,.2f} (±{cv_scores_df['train_mae'].std():,.2f})")
        print(f"평균 Val MAE:    {cv_scores_df['val_mae'].mean():,.2f} (±{cv_scores_df['val_mae'].std():,.2f})")
        
        # OOF 전체 성능
        oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        oof_mae = mean_absolute_error(y, oof_predictions)
        print(f"\nOOF RMSE: {oof_rmse:,.2f}")
        print(f"OOF MAE:  {oof_mae:,.2f}")
    
    return {
        'scores': cv_scores_df,
        'oof_predictions': oof_predictions,
        'models': models,
        'mean_val_rmse': cv_scores_df['val_rmse'].mean(),
        'mean_val_mae': cv_scores_df['val_mae'].mean(),
        'std_val_rmse': cv_scores_df['val_rmse'].std(),
        'std_val_mae': cv_scores_df['val_mae'].std()
    }


def get_oof_predictions(
    model_class,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 3,
    model_params: Dict[str, Any] = None
) -> np.ndarray:
    """
    Out-of-Fold 예측값만 빠르게 반환
    
    Args:
        model_class: 모델 클래스
        X: 특징 데이터
        y: 타겟 데이터
        n_splits: CV fold 수
        test_size: validation 크기
        model_params: 모델 파라미터
    
    Returns:
        OOF 예측값 배열
    """
    result = cross_validate_model(
        model_class=model_class,
        X=X,
        y=y,
        n_splits=n_splits,
        test_size=test_size,
        model_params=model_params,
        verbose=False
    )
    return result['oof_predictions']


if __name__ == "__main__":
    # 테스트 코드
    print("Forward Chaining CV 시스템 테스트\n")
    
    # 더미 데이터 생성 (43개월 시뮬레이션)
    np.random.seed(42)
    n_samples = 43
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    y = pd.Series(np.random.randn(n_samples) * 100 + 1000)
    
    # CV 시스템 테스트
    cv = ForwardChainingCV(n_splits=5, test_size=3)
    
    print("Split 정보:")
    split_info = cv.get_split_info(X)
    print(split_info)
    print("\n✅ 모든 Fold에서 Data Leakage 없음!" if split_info['no_leakage'].all() else "❌ Data Leakage 발견!")
    
    # LightGBM 테스트 (설치되어 있는 경우)
    try:
        from lightgbm import LGBMRegressor
        
        print("\n" + "="*50)
        print("LightGBM Forward Chaining CV 테스트")
        print("="*50)
        
        result = cross_validate_model(
            model_class=LGBMRegressor,
            X=X,
            y=y,
            n_splits=5,
            test_size=3,
            model_params={'n_estimators': 100, 'random_state': 42, 'verbose': -1}
        )
        
        print("\n✅ Forward Chaining CV 완료!")
        print(f"평균 Val RMSE: {result['mean_val_rmse']:.2f} (±{result['std_val_rmse']:.2f})")
        
    except ImportError:
        print("\n⚠️ LightGBM이 설치되어 있지 않아 테스트를 건너뜁니다.")
