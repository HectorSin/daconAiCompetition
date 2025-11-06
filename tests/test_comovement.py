"""
공행성 탐지 함수에 대한 단위 테스트
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.comovement import (
    calculate_ccf,
    calculate_ccf_matrix,
    calculate_granger_causality,
    calculate_granger_matrix,
    calculate_dtw_distance,
    calculate_dtw_matrix,
    apply_fdr_correction,
    apply_fdr_to_results,
    detect_comovement_comprehensive
)


@pytest.fixture
def simple_series():
    """간단한 시계열 생성"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    x = np.random.randn(50)
    return pd.Series(x, index=dates)


@pytest.fixture
def lagged_series():
    """시차가 있는 시계열 쌍 생성"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    x = np.random.randn(50)
    # y는 x를 2개월 지연시킨 것 + 노이즈
    y = np.concatenate([np.zeros(2), x[:-2]]) + np.random.randn(50) * 0.1
    return pd.Series(x, index=dates), pd.Series(y, index=dates)


@pytest.fixture
def similar_series():
    """유사한 패턴의 시계열 쌍 생성"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    t = np.arange(50)
    x = 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(50)
    y = 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(50)
    return pd.Series(x, index=dates), pd.Series(y, index=dates)


@pytest.fixture
def multi_item_df():
    """여러 품목의 데이터프레임 생성"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')

    # 의도적인 선행-후행 관계 생성
    item_00 = np.random.randn(50)
    item_01 = np.concatenate([np.zeros(2), item_00[:-2]]) + np.random.randn(50) * 0.2
    item_02 = np.random.randn(50)  # 독립적

    df = pd.DataFrame({
        'item_00': item_00,
        'item_01': item_01,
        'item_02': item_02
    }, index=dates)

    return df


def test_calculate_ccf(lagged_series):
    """CCF 계산 테스트"""
    x, y = lagged_series
    ccf_values, optimal_lag, max_ccf = calculate_ccf(x, y, max_lag=6)

    assert isinstance(ccf_values, np.ndarray)
    assert isinstance(optimal_lag, (int, np.integer))
    assert isinstance(max_ccf, (float, np.floating))
    assert 0 <= optimal_lag <= 6
    assert -1 <= max_ccf <= 1


def test_calculate_ccf_matrix(multi_item_df):
    """CCF 행렬 계산 테스트"""
    results = calculate_ccf_matrix(multi_item_df, max_lag=6, threshold=0.3)

    assert isinstance(results, pd.DataFrame)
    assert 'item_x' in results.columns
    assert 'item_y' in results.columns
    assert 'optimal_lag' in results.columns
    assert 'max_ccf' in results.columns
    assert 'is_significant' in results.columns
    # 3개 품목이므로 3개 쌍 (0-1, 0-2, 1-2)
    assert len(results) == 3


def test_calculate_granger_causality(lagged_series):
    """Granger 인과관계 계산 테스트"""
    x, y = lagged_series
    result = calculate_granger_causality(x, y, max_lag=6)

    assert 'best_lag' in result
    assert 'best_p_value' in result
    assert 'is_causal' in result
    assert isinstance(result['is_causal'], bool)


def test_calculate_granger_matrix(multi_item_df):
    """Granger 인과관계 행렬 계산 테스트"""
    results = calculate_granger_matrix(multi_item_df, max_lag=6)

    assert isinstance(results, pd.DataFrame)
    assert 'item_cause' in results.columns
    assert 'item_effect' in results.columns
    assert 'best_lag' in results.columns
    assert 'p_value' in results.columns
    assert 'is_causal' in results.columns
    # 3개 품목, 양방향이므로 6개 쌍
    assert len(results) == 6


def test_calculate_dtw_distance(similar_series):
    """DTW 거리 계산 테스트"""
    x, y = similar_series
    distance = calculate_dtw_distance(x, y)

    assert isinstance(distance, (float, np.floating))
    assert distance >= 0


def test_calculate_dtw_matrix(multi_item_df):
    """DTW 행렬 계산 테스트"""
    results = calculate_dtw_matrix(multi_item_df, threshold=0.5)

    assert isinstance(results, pd.DataFrame)
    assert 'item_x' in results.columns
    assert 'item_y' in results.columns
    assert 'dtw_distance' in results.columns
    assert 'is_similar' in results.columns
    # 3개 품목이므로 3개 쌍
    assert len(results) == 3


def test_apply_fdr_correction():
    """FDR 보정 테스트"""
    # 테스트 p-value 생성
    p_values = np.array([0.001, 0.01, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 0.9])
    rejected, threshold = apply_fdr_correction(p_values, alpha=0.05)

    assert isinstance(rejected, np.ndarray)
    assert len(rejected) == len(p_values)
    assert rejected.dtype == bool
    assert isinstance(threshold, (float, np.floating))


def test_apply_fdr_to_results():
    """결과 데이터프레임에 FDR 보정 적용 테스트"""
    # 테스트 데이터프레임 생성
    df = pd.DataFrame({
        'item_x': ['A', 'B', 'C'],
        'item_y': ['B', 'C', 'A'],
        'p_value': [0.01, 0.03, 0.5]
    })

    results = apply_fdr_to_results(df, p_value_col='p_value', alpha=0.05)

    assert 'fdr_rejected' in results.columns
    assert 'fdr_threshold' in results.columns
    assert len(results) == len(df)


def test_detect_comovement_comprehensive(multi_item_df):
    """종합 공행성 탐지 테스트"""
    results = detect_comovement_comprehensive(
        multi_item_df,
        ccf_threshold=0.3,
        granger_alpha=0.05,
        dtw_threshold=0.5,
        max_lag=6,
        apply_fdr=True,
        fdr_alpha=0.05
    )

    assert 'ccf' in results
    assert 'granger' in results
    assert 'dtw' in results
    assert isinstance(results['ccf'], pd.DataFrame)
    assert isinstance(results['granger'], pd.DataFrame)
    assert isinstance(results['dtw'], pd.DataFrame)


def test_ccf_with_missing_values():
    """결측값이 있는 시계열에 대한 CCF 테스트"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    x = np.random.randn(50)
    y = np.random.randn(50)
    x[10:15] = np.nan

    x_series = pd.Series(x, index=dates)
    y_series = pd.Series(y, index=dates)

    ccf_values, optimal_lag, max_ccf = calculate_ccf(x_series, y_series, max_lag=6)

    assert isinstance(ccf_values, np.ndarray)
    # 결측값이 제거된 후에도 계산 가능해야 함


def test_fdr_correction_with_all_significant():
    """모두 유의한 p-value에 대한 FDR 보정"""
    p_values = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    rejected, threshold = apply_fdr_correction(p_values, alpha=0.05)

    # 모두 유의해야 함
    assert rejected.all()


def test_fdr_correction_with_none_significant():
    """모두 비유의한 p-value에 대한 FDR 보정"""
    p_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    rejected, threshold = apply_fdr_correction(p_values, alpha=0.05)

    # 모두 비유의해야 함
    assert not rejected.any()
