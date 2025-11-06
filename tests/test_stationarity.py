"""
정상성 테스트 함수에 대한 단위 테스트
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import (
    check_stationarity_adf,
    check_stationarity_kpss,
    check_stationarity_all_items,
    decompose_stl,
    decompose_all_items
)


@pytest.fixture
def stationary_series():
    """정상 시계열 생성 (백색 잡음)"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    values = np.random.randn(50)
    return pd.Series(values, index=dates)


@pytest.fixture
def non_stationary_series():
    """비정상 시계열 생성 (랜덤 워크)"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    values = np.cumsum(np.random.randn(50))
    return pd.Series(values, index=dates)


@pytest.fixture
def seasonal_series():
    """계절성이 있는 시계열 생성"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=48, freq='ME')
    t = np.arange(48)
    trend = t * 0.5
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(48) * 2
    values = trend + seasonal + noise
    return pd.Series(values, index=dates)


def test_adf_on_stationary_series(stationary_series):
    """정상 시계열에 대한 ADF 테스트"""
    result = check_stationarity_adf(stationary_series)

    assert result['test'] == 'ADF'
    assert 'p_value' in result
    assert 'is_stationary' in result
    assert isinstance(result['is_stationary'], bool)


def test_adf_on_non_stationary_series(non_stationary_series):
    """비정상 시계열에 대한 ADF 테스트"""
    result = check_stationarity_adf(non_stationary_series)

    assert result['test'] == 'ADF'
    assert 'p_value' in result
    # 비정상 시계열이므로 p-value가 높을 것으로 예상
    # (단, 항상 보장되는 것은 아님)


def test_kpss_on_stationary_series(stationary_series):
    """정상 시계열에 대한 KPSS 테스트"""
    result = check_stationarity_kpss(stationary_series)

    assert result['test'] == 'KPSS'
    assert 'p_value' in result
    assert 'is_stationary' in result
    assert isinstance(result['is_stationary'], bool)


def test_stationarity_all_items():
    """모든 품목에 대한 정상성 테스트"""
    # 테스트 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')

    df_wide = pd.DataFrame({
        'item_00': np.random.randn(50),  # 정상
        'item_01': np.cumsum(np.random.randn(50)),  # 비정상
        'item_02': np.random.randn(50),  # 정상
    }, index=dates)

    results = check_stationarity_all_items(df_wide)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 3
    assert 'item_code' in results.columns
    assert 'adf_p_value' in results.columns
    assert 'kpss_p_value' in results.columns
    assert 'both_stationary' in results.columns


def test_decompose_stl(seasonal_series):
    """STL 분해 테스트"""
    trend, seasonal, resid = decompose_stl(seasonal_series, period=12)

    assert isinstance(trend, pd.Series)
    assert isinstance(seasonal, pd.Series)
    assert isinstance(resid, pd.Series)
    assert len(trend) == len(seasonal_series)


def test_decompose_all_items():
    """모든 품목에 대한 STL 분해 테스트"""
    # 테스트 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=48, freq='ME')
    t = np.arange(48)

    df_wide = pd.DataFrame({
        'item_00': 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(48),
        'item_01': 5 * np.sin(2 * np.pi * t / 12) + np.random.randn(48),
    }, index=dates)

    results = decompose_all_items(df_wide, period=12)

    assert 'trend' in results
    assert 'seasonal' in results
    assert 'resid' in results
    assert isinstance(results['trend'], pd.DataFrame)
    assert isinstance(results['seasonal'], pd.DataFrame)
    assert isinstance(results['resid'], pd.DataFrame)


def test_adf_with_missing_values():
    """결측값이 있는 시계열에 대한 ADF 테스트"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    values = np.random.randn(50)
    values[10:15] = np.nan

    series = pd.Series(values, index=dates)
    result = check_stationarity_adf(series)

    assert 'p_value' in result
    assert not np.isnan(result['p_value'])
