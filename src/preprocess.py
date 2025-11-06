"""
데이터 전처리 모듈

무역 데이터 로드, 변환, 품질 체크 기능을 제공합니다.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

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
