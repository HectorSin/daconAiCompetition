"""
데이터 전처리 모듈

무역 데이터 로드, 변환, 품질 체크 기능을 제공합니다.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: Path) -> pd.DataFrame:
    """
    CSV 파일에서 데이터를 로드합니다.

    Parameters:
    -----------
    file_path : Path
        로드할 CSV 파일 경로

    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임 (long format)
    """
    logger.info(f"데이터 로딩 중: {file_path}")

    df = pd.read_csv(file_path)

    # 날짜 컬럼을 datetime으로 변환
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    logger.info(f"데이터 로드 완료: {df.shape[0]} rows, {df['item_code'].nunique() if 'item_code' in df.columns else 'N/A'} items")

    return df


def pivot_data(df: pd.DataFrame,
               date_col: str = 'date',
               item_col: str = 'item_code',
               value_col: str = 'value') -> pd.DataFrame:
    """
    Long format 데이터를 Wide format으로 변환합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        Long format 데이터프레임
    date_col : str
        날짜 컬럼명
    item_col : str
        품목 컬럼명
    value_col : str
        값 컬럼명

    Returns:
    --------
    pd.DataFrame
        Wide format 데이터프레임 (index: date, columns: item_code)
    """
    logger.info(f"데이터 피벗 중: {date_col} x {item_col}")

    df_wide = df.pivot(index=date_col, columns=item_col, values=value_col)

    logger.info(f"피벗 완료: {df_wide.shape} (날짜 x 품목)")

    return df_wide


def check_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    음수 값을 체크하고 로깅합니다.

    무역량은 0 이상이어야 하므로 음수가 있으면 경고를 발생시킵니다.

    Parameters:
    -----------
    df : pd.DataFrame
        체크할 데이터프레임 (wide format)

    Returns:
    --------
    pd.DataFrame
        원본 데이터프레임 (변경 없음)
    """
    negative_mask = df < 0
    negative_count = negative_mask.sum().sum()

    if negative_count > 0:
        logger.warning(f"⚠️  음수 값 발견: {negative_count}개")

        # 음수가 있는 컬럼 출력
        cols_with_negatives = negative_mask.sum()[negative_mask.sum() > 0]
        logger.warning(f"음수 포함 품목: {list(cols_with_negatives.index)}")
    else:
        logger.info("✓ 음수 값 없음")

    return df


def log_outliers(df: pd.DataFrame,
                iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    IQR 방법을 사용하여 이상치를 탐지하고 로깅합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        체크할 데이터프레임 (wide format)
    iqr_multiplier : float
        IQR 승수 (기본값: 1.5)

    Returns:
    --------
    pd.DataFrame
        원본 데이터프레임 (변경 없음)
    """
    logger.info(f"이상치 탐지 중 (IQR * {iqr_multiplier})...")

    outlier_summary = []

    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

        if len(outliers) > 0:
            outlier_summary.append({
                'item_code': col,
                'n_outliers': len(outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_outlier': outliers.min(),
                'max_outlier': outliers.max()
            })

    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        logger.warning(f"⚠️  이상치 발견: {len(outlier_summary)}개 품목")
        logger.info(f"\n{outlier_df.to_string()}")
    else:
        logger.info("✓ 이상치 없음")

    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측값을 체크하고 로깅합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        체크할 데이터프레임

    Returns:
    --------
    pd.DataFrame
        원본 데이터프레임 (변경 없음)
    """
    missing_count = df.isnull().sum()
    missing_items = missing_count[missing_count > 0]

    if len(missing_items) > 0:
        logger.warning(f"⚠️  결측값 발견: {len(missing_items)}개 품목")
        logger.info(f"\n{missing_items.to_string()}")
    else:
        logger.info("✓ 결측값 없음")

    return df


def preprocess_pipeline(file_path: Path,
                       check_quality: bool = True) -> pd.DataFrame:
    """
    전체 전처리 파이프라인을 실행합니다.

    Parameters:
    -----------
    file_path : Path
        데이터 파일 경로
    check_quality : bool
        데이터 품질 체크 수행 여부

    Returns:
    --------
    pd.DataFrame
        전처리된 데이터프레임 (wide format)
    """
    # 1. 데이터 로드
    df_long = load_data(file_path)

    # 2. Wide format 변환
    df_wide = pivot_data(df_long)

    # 3. 데이터 품질 체크 (선택적)
    if check_quality:
        logger.info("\n=== 데이터 품질 체크 ===")
        check_missing_values(df_wide)
        check_negative_values(df_wide)
        log_outliers(df_wide)

    return df_wide


def check_stationarity_adf(series: pd.Series,
                          significance_level: float = 0.05) -> Dict:
    """
    ADF (Augmented Dickey-Fuller) 검정으로 정상성을 테스트합니다.

    귀무가설(H0): 시계열이 비정상성(단위근 존재)
    p-value < 0.05 이면 귀무가설 기각 -> 정상성

    Parameters:
    -----------
    series : pd.Series
        테스트할 시계열 데이터
    significance_level : float
        유의수준 (기본값: 0.05)

    Returns:
    --------
    Dict
        테스트 결과 딕셔너리
    """
    result = adfuller(series.dropna())

    return {
        'test': 'ADF',
        'statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': bool(result[1] < significance_level)
    }


def check_stationarity_kpss(series: pd.Series,
                           significance_level: float = 0.05,
                           regression: str = 'c') -> Dict:
    """
    KPSS (Kwiatkowski-Phillips-Schmidt-Shin) 검정으로 정상성을 테스트합니다.

    귀무가설(H0): 시계열이 정상성
    p-value < 0.05 이면 귀무가설 기각 -> 비정상성

    Parameters:
    -----------
    series : pd.Series
        테스트할 시계열 데이터
    significance_level : float
        유의수준 (기본값: 0.05)
    regression : str
        'c' (상수) 또는 'ct' (상수+추세)

    Returns:
    --------
    Dict
        테스트 결과 딕셔너리
    """
    result = kpss(series.dropna(), regression=regression, nlags='auto')

    return {
        'test': 'KPSS',
        'statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'critical_values': result[3],
        'is_stationary': bool(result[1] >= significance_level)
    }


def check_stationarity_all_items(df_wide: pd.DataFrame,
                                 significance_level: float = 0.05) -> pd.DataFrame:
    """
    모든 품목에 대해 ADF와 KPSS 정상성 검정을 수행합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터프레임 (index: date, columns: item_code)
    significance_level : float
        유의수준 (기본값: 0.05)

    Returns:
    --------
    pd.DataFrame
        각 품목의 정상성 테스트 결과
    """
    logger.info(f"정상성 테스트 수행 중 ({len(df_wide.columns)}개 품목)...")

    results = []

    for item_code in df_wide.columns:
        series = df_wide[item_code]

        # ADF 테스트
        adf_result = check_stationarity_adf(series, significance_level)

        # KPSS 테스트
        kpss_result = check_stationarity_kpss(series, significance_level)

        # 결과 요약
        # ADF와 KPSS 모두 정상성을 나타내야 확실한 정상성
        results.append({
            'item_code': item_code,
            'adf_statistic': adf_result['statistic'],
            'adf_p_value': adf_result['p_value'],
            'adf_is_stationary': adf_result['is_stationary'],
            'kpss_statistic': kpss_result['statistic'],
            'kpss_p_value': kpss_result['p_value'],
            'kpss_is_stationary': kpss_result['is_stationary'],
            'both_stationary': adf_result['is_stationary'] and kpss_result['is_stationary']
        })

    results_df = pd.DataFrame(results)

    # 요약 통계
    n_adf_stationary = results_df['adf_is_stationary'].sum()
    n_kpss_stationary = results_df['kpss_is_stationary'].sum()
    n_both_stationary = results_df['both_stationary'].sum()

    logger.info(f"\n정상성 테스트 결과:")
    logger.info(f"  ADF 정상성: {n_adf_stationary}/{len(results_df)} 품목")
    logger.info(f"  KPSS 정상성: {n_kpss_stationary}/{len(results_df)} 품목")
    logger.info(f"  양쪽 모두 정상성: {n_both_stationary}/{len(results_df)} 품목")

    return results_df


def decompose_stl(series: pd.Series,
                 period: int = 12,
                 seasonal: int = 7) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    STL (Seasonal-Trend decomposition using Loess) 분해를 수행합니다.

    Parameters:
    -----------
    series : pd.Series
        분해할 시계열 데이터 (index는 datetime)
    period : int
        계절 주기 (기본값: 12 - 월별 데이터)
    seasonal : int
        계절 성분 평활화 창 크기 (홀수, 기본값: 7)

    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.Series]
        (trend, seasonal, residual) 시계열
    """
    # STL 분해 수행
    stl = STL(series.dropna(), period=period, seasonal=seasonal)
    result = stl.fit()

    return result.trend, result.seasonal, result.resid


def decompose_all_items(df_wide: pd.DataFrame,
                       period: int = 12,
                       seasonal: int = 7) -> Dict[str, pd.DataFrame]:
    """
    모든 품목에 대해 STL 분해를 수행합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터프레임
    period : int
        계절 주기 (기본값: 12)
    seasonal : int
        계절 성분 평활화 창 크기 (기본값: 7)

    Returns:
    --------
    Dict[str, pd.DataFrame]
        {'trend': df_trend, 'seasonal': df_seasonal, 'resid': df_resid}
    """
    logger.info(f"STL 분해 수행 중 ({len(df_wide.columns)}개 품목)...")

    trends = {}
    seasonals = {}
    resids = {}

    for item_code in df_wide.columns:
        series = df_wide[item_code]

        try:
            trend, seasonal, resid = decompose_stl(series, period, seasonal)
            trends[item_code] = trend
            seasonals[item_code] = seasonal
            resids[item_code] = resid
        except Exception as e:
            logger.warning(f"품목 {item_code} STL 분해 실패: {e}")
            continue

    logger.info(f"✓ STL 분해 완료: {len(trends)}개 품목")

    return {
        'trend': pd.DataFrame(trends),
        'seasonal': pd.DataFrame(seasonals),
        'resid': pd.DataFrame(resids)
    }
