"""
특징 공학 모듈

시계열 특징을 생성하여 예측 모델의 입력으로 사용합니다.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_lag_features(df: pd.DataFrame,
                          target_col: str,
                          lags: List[int],
                          prefix: str = '') -> pd.DataFrame:
    """
    Lag 특징을 생성합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터프레임 (index는 datetime)
    target_col : str
        Lag를 생성할 컬럼명
    lags : List[int]
        생성할 lag 리스트 (예: [1, 2, 3, 6, 12])
    prefix : str
        특징명 접두사 (기본값: '')

    Returns:
    --------
    pd.DataFrame
        Lag 특징이 추가된 데이터프레임
    """
    result = df.copy()

    for lag in lags:
        col_name = f'{prefix}{target_col}_lag_{lag}' if prefix else f'{target_col}_lag_{lag}'
        result[col_name] = df[target_col].shift(lag)

    return result


def generate_rolling_features(df: pd.DataFrame,
                              target_col: str,
                              windows: List[int],
                              prefix: str = '') -> pd.DataFrame:
    """
    Rolling 통계 특징을 생성합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터프레임
    target_col : str
        Rolling 통계를 계산할 컬럼명
    windows : List[int]
        윈도우 크기 리스트 (예: [3, 6, 12])
    prefix : str
        특징명 접두사

    Returns:
    --------
    pd.DataFrame
        Rolling 특징이 추가된 데이터프레임
    """
    result = df.copy()

    for window in windows:
        # Rolling mean
        col_name = f'{prefix}{target_col}_rolling_mean_{window}' if prefix else f'{target_col}_rolling_mean_{window}'
        result[col_name] = df[target_col].rolling(window=window).mean()

        # Rolling std
        col_name = f'{prefix}{target_col}_rolling_std_{window}' if prefix else f'{target_col}_rolling_std_{window}'
        result[col_name] = df[target_col].rolling(window=window).std()

        # Rolling min
        col_name = f'{prefix}{target_col}_rolling_min_{window}' if prefix else f'{target_col}_rolling_min_{window}'
        result[col_name] = df[target_col].rolling(window=window).min()

        # Rolling max
        col_name = f'{prefix}{target_col}_rolling_max_{window}' if prefix else f'{target_col}_rolling_max_{window}'
        result[col_name] = df[target_col].rolling(window=window).max()

    return result


def generate_growth_rate_features(df: pd.DataFrame,
                                  target_col: str,
                                  periods: List[int] = [1, 12],
                                  prefix: str = '') -> pd.DataFrame:
    """
    성장률 특징을 생성합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터프레임
    target_col : str
        성장률을 계산할 컬럼명
    periods : List[int]
        비교 기간 리스트 (예: [1, 12] - MoM, YoY)
    prefix : str
        특징명 접두사

    Returns:
    --------
    pd.DataFrame
        성장률 특징이 추가된 데이터프레임
    """
    result = df.copy()

    for period in periods:
        # 변화량
        col_name = f'{prefix}{target_col}_diff_{period}' if prefix else f'{target_col}_diff_{period}'
        result[col_name] = df[target_col].diff(period)

        # 증가율 (%)
        col_name = f'{prefix}{target_col}_pct_change_{period}' if prefix else f'{target_col}_pct_change_{period}'
        result[col_name] = df[target_col].pct_change(period) * 100

    return result


def generate_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜 관련 특징을 생성합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터프레임 (index는 datetime)

    Returns:
    --------
    pd.DataFrame
        날짜 특징이 추가된 데이터프레임
    """
    result = df.copy()

    # 월
    result['month'] = result.index.month

    # 분기
    result['quarter'] = result.index.quarter

    # 연도
    result['year'] = result.index.year

    # 월별 사인/코사인 변환 (순환성 표현)
    result['month_sin'] = np.sin(2 * np.pi * result.index.month / 12)
    result['month_cos'] = np.cos(2 * np.pi * result.index.month / 12)

    # 분기별 사인/코사인 변환
    result['quarter_sin'] = np.sin(2 * np.pi * result.index.quarter / 4)
    result['quarter_cos'] = np.cos(2 * np.pi * result.index.quarter / 4)

    return result


def generate_leading_item_features(df_wide: pd.DataFrame,
                                   target_item: str,
                                   ccf_results: pd.DataFrame,
                                   lags: List[int] = [1, 2, 3],
                                   top_n: int = 5,
                                   threshold: float = 0.5) -> pd.DataFrame:
    """
    선행 품목의 특징을 생성합니다 (공행성 활용).

    이 함수가 이번 대회의 핵심입니다!
    타겟 품목의 선행 품목들의 과거 값을 특징으로 사용합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터 (날짜 x 품목)
    target_item : str
        타겟 품목 (후행 품목)
    ccf_results : pd.DataFrame
        CCF 분석 결과 (item_x, item_y, optimal_lag, max_ccf 포함)
    lags : List[int]
        생성할 lag 리스트
    top_n : int
        상위 몇 개의 선행 품목을 사용할지
    threshold : float
        CCF 임계값 (이상인 것만 선택)

    Returns:
    --------
    pd.DataFrame
        선행 품목 특징이 추가된 데이터프레임
    """
    result = pd.DataFrame(index=df_wide.index)

    # 타겟 품목이 후행 품목인 쌍 찾기
    leading_pairs = ccf_results[
        (ccf_results['item_y'] == target_item) &
        (ccf_results['abs_ccf'] >= threshold)
    ].nlargest(top_n, 'abs_ccf')

    if len(leading_pairs) == 0:
        logger.warning(f"품목 {target_item}에 대한 선행 품목이 없습니다.")
        return result

    logger.info(f"품목 {target_item}에 대한 선행 품목 {len(leading_pairs)}개 발견")

    # 각 선행 품목에 대해 특징 생성
    for idx, row in leading_pairs.iterrows():
        leading_item = row['item_x']
        optimal_lag = int(row['optimal_lag'])
        ccf_value = row['max_ccf']

        # 선행 품목의 값을 최적 시차만큼 shift
        for lag in lags:
            total_lag = optimal_lag + lag
            col_name = f'leading_{leading_item}_lag_{total_lag}_ccf_{ccf_value:.2f}'

            if leading_item in df_wide.columns:
                result[col_name] = df_wide[leading_item].shift(total_lag)

    return result


def generate_interaction_features(df: pd.DataFrame,
                                  base_features: List[str],
                                  max_interactions: int = 5) -> pd.DataFrame:
    """
    상호작용 특징을 생성합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터프레임
    base_features : List[str]
        상호작용을 생성할 기본 특징 리스트
    max_interactions : int
        생성할 최대 상호작용 개수

    Returns:
    --------
    pd.DataFrame
        상호작용 특징이 추가된 데이터프레임
    """
    result = df.copy()

    # 간단한 상호작용만 생성 (비율)
    if len(base_features) >= 2:
        for i in range(min(len(base_features), max_interactions)):
            for j in range(i+1, min(len(base_features), max_interactions)):
                feat1 = base_features[i]
                feat2 = base_features[j]

                # 비율
                col_name = f'{feat1}_div_{feat2}'
                result[col_name] = df[feat1] / (df[feat2] + 1e-8)  # 0으로 나누기 방지

    return result


def create_features_for_item(df_wide: pd.DataFrame,
                             target_item: str,
                             ccf_results: pd.DataFrame,
                             lag_features: List[int] = [1, 2, 3, 6, 12],
                             rolling_windows: List[int] = [3, 6, 12],
                             include_leading: bool = True,
                             top_leading: int = 5) -> pd.DataFrame:
    """
    특정 품목에 대한 모든 특징을 생성합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터 (날짜 x 품목)
    target_item : str
        타겟 품목
    ccf_results : pd.DataFrame
        CCF 분석 결과
    lag_features : List[int]
        생성할 lag 리스트
    rolling_windows : List[int]
        Rolling 윈도우 크기 리스트
    include_leading : bool
        선행 품목 특징 포함 여부
    top_leading : int
        사용할 상위 선행 품목 개수

    Returns:
    --------
    pd.DataFrame
        모든 특징이 포함된 데이터프레임
    """
    logger.info(f"품목 {target_item}에 대한 특징 생성 중...")

    # 타겟 품목 데이터 추출
    df_item = pd.DataFrame({target_item: df_wide[target_item]})

    # 1. Lag 특징
    df_features = generate_lag_features(df_item, target_item, lag_features)

    # 2. Rolling 특징
    df_features = generate_rolling_features(df_features, target_item, rolling_windows)

    # 3. 성장률 특징
    df_features = generate_growth_rate_features(df_features, target_item, periods=[1, 12])

    # 4. 날짜 특징
    df_features = generate_date_features(df_features)

    # 5. 선행 품목 특징 (공행성 활용)
    if include_leading:
        df_leading = generate_leading_item_features(
            df_wide, target_item, ccf_results,
            lags=[0, 1, 2],  # 선행 품목의 추가 lag
            top_n=top_leading
        )
        df_features = pd.concat([df_features, df_leading], axis=1)

    logger.info(f"특징 생성 완료: {df_features.shape[1]}개 특징")

    return df_features


def create_features_for_all_items(df_wide: pd.DataFrame,
                                  ccf_results: pd.DataFrame,
                                  lag_features: List[int] = [1, 2, 3, 6],
                                  rolling_windows: List[int] = [3, 6],
                                  include_leading: bool = True,
                                  top_leading: int = 3,
                                  save_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    모든 품목에 대한 특징을 생성합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터
    ccf_results : pd.DataFrame
        CCF 분석 결과
    lag_features : List[int]
        Lag 리스트
    rolling_windows : List[int]
        Rolling 윈도우 리스트
    include_leading : bool
        선행 품목 특징 포함 여부
    top_leading : int
        상위 선행 품목 개수
    save_path : Optional[Path]
        저장 경로 (None이면 저장 안 함)

    Returns:
    --------
    Dict[str, pd.DataFrame]
        {품목명: 특징 데이터프레임} 딕셔너리
    """
    logger.info(f"전체 {len(df_wide.columns)}개 품목에 대한 특징 생성 시작...")

    all_features = {}

    for item in df_wide.columns:
        features = create_features_for_item(
            df_wide, item, ccf_results,
            lag_features=lag_features,
            rolling_windows=rolling_windows,
            include_leading=include_leading,
            top_leading=top_leading
        )

        all_features[item] = features

        # 저장
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            features.to_csv(save_path / f'features_{item}.csv')

    logger.info(f"✓ 전체 특징 생성 완료")

    return all_features


def prepare_train_data(features_df: pd.DataFrame,
                       target_col: str,
                       min_samples: int = 12) -> Tuple[pd.DataFrame, pd.Series]:
    """
    학습 데이터를 준비합니다 (결측값 제거).

    Parameters:
    -----------
    features_df : pd.DataFrame
        특징 데이터프레임
    target_col : str
        타겟 컬럼명
    min_samples : int
        최소 샘플 개수

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) - 특징과 타겟
    """
    # 타겟 컬럼 분리
    y = features_df[target_col].copy()
    X = features_df.drop(columns=[target_col])

    # 결측값이 있는 행 제거
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]

    if len(X_clean) < min_samples:
        logger.warning(f"유효한 샘플이 {len(X_clean)}개로 부족합니다 (최소 {min_samples}개 필요)")

    logger.info(f"학습 데이터 준비 완료: {X_clean.shape[0]}개 샘플, {X_clean.shape[1]}개 특징")

    return X_clean, y_clean
